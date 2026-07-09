import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from src.dataloaders.protein import ProteinDataset, ProteinTokenizer, create_dataloader
except ImportError:
    from models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from dataloaders.protein import ProteinDataset, ProteinTokenizer, create_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train S4 protein model")
    p.add_argument("--fasta_file", type=str, default="data/protein/uniref50.fasta.gz",
                   help="Path to training FASTA (plain or .gz; UniRef50 only)")
    p.add_argument("--val_fasta", type=str, default=None, help="Validation FASTA (optional; use split or holdout from same file)")
    p.add_argument("--l_max", type=int, default=1024)
    p.add_argument("--objective", type=str, default="autocomplete", choices=["autocomplete", "masked"],
                   help="Training objective: prefix autocomplete or legacy masked LM")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--prefix_length", type=int, default=None,
                   help="Fixed autocomplete prefix length. If omitted, use random prefix fractions.")
    p.add_argument("--prefix_min_fraction", type=float, default=0.25)
    p.add_argument("--prefix_max_fraction", type=float, default=0.70)
    p.add_argument("--end_prefix_prob", type=float, default=0.0,
                   help="Probability of sampling autocomplete prefixes near sequence ends.")
    p.add_argument("--end_prefix_min_fraction", type=float, default=0.75)
    p.add_argument("--end_prefix_max_fraction", type=float, default=0.95)
    p.add_argument("--eos_loss_weight", type=float, default=1.0,
                   help="Extra class weight for EOS targets in autocomplete loss.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--kernel_type", type=str, default="diag", choices=["diag", "nplr"])
    p.add_argument("--bidirectional", dest="bidirectional", action="store_true", default=None,
                   help="Force bidirectional S4 blocks")
    p.add_argument("--unidirectional", dest="bidirectional", action="store_false",
                   help="Force unidirectional S4 blocks")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--model_size", type=str, default=None, choices=["small", "base", "large"],
                   help="Override with preset small/base/large")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--save_best", action="store_true", default=True)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Maximum number of sequences to load (for memory-limited systems). "
                        "E.g., --max_sequences 1000000 for 1M sequences. None = load all.")
    return p.parse_args()


def unpack_batch(batch, device):
    if len(batch) == 3:
        input_ids, target_ids, attention_mask = batch
        loss_mask = None
    elif len(batch) == 4:
        input_ids, target_ids, attention_mask, loss_mask = batch
        loss_mask = loss_mask.to(device)
    else:
        raise ValueError(f"Expected 3 or 4 tensors per batch, got {len(batch)}")

    return (
        input_ids.to(device),
        target_ids.to(device),
        attention_mask.to(device),
        loss_mask,
    )


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args, global_step, use_amp):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Train")
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(pbar):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss = model.compute_loss(
                    input_ids, target_ids, attention_mask,
                    loss_mask=loss_mask, objective=args.objective,
                    eos_loss_weight=args.eos_loss_weight,
                )
                if args.grad_accum_steps > 1:
                    loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
        else:
            loss = model.compute_loss(
                input_ids, target_ids, attention_mask,
                loss_mask=loss_mask, objective=args.objective,
                eos_loss_weight=args.eos_loss_weight,
            )
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

        total_loss += loss.item() * (args.grad_accum_steps if args.grad_accum_steps > 1 else 1)
        n_batches += 1
        pbar.set_postfix(loss=loss.item() * (args.grad_accum_steps if args.grad_accum_steps > 1 else 1))

    if args.grad_accum_steps > 1 and n_batches % args.grad_accum_steps != 0:
        if use_amp and device.type == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        global_step += 1

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate(model, loader, device, objective, eos_loss_weight=1.0):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val"):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)
        loss = model.compute_loss(
            input_ids, target_ids, attention_mask,
            loss_mask=loss_mask, objective=objective,
            eos_loss_weight=eos_loss_weight,
        )
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device()

    tokenizer = ProteinTokenizer()
    bidirectional = args.bidirectional
    if bidirectional is None:
        bidirectional = args.objective == "masked"
    
    print(f"🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   FASTA: {args.fasta_file}")
    print(f"   Objective: {args.objective}")
    print(f"   Bidirectional: {bidirectional}")
    if args.max_sequences:
        print(f"   Max sequences: {args.max_sequences:,}")
    if args.objective == "autocomplete":
        if args.prefix_length:
            print(f"   Prefix length: {args.prefix_length}")
        else:
            print(f"   Prefix fraction: {args.prefix_min_fraction:.2f}-{args.prefix_max_fraction:.2f}")
        if args.end_prefix_prob > 0:
            print(
                f"   End-prefix sampling: p={args.end_prefix_prob:.2f}, "
                f"fraction={args.end_prefix_min_fraction:.2f}-{args.end_prefix_max_fraction:.2f}"
            )
        if args.eos_loss_weight != 1.0:
            print(f"   EOS loss weight: {args.eos_loss_weight:.2f}")
    print()
    
    train_dataset = ProteinDataset(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        l_max=args.l_max,
        mask_prob=args.mask_prob,
        objective=args.objective,
        prefix_length=args.prefix_length,
        prefix_min_fraction=args.prefix_min_fraction,
        prefix_max_fraction=args.prefix_max_fraction,
        end_prefix_prob=args.end_prefix_prob,
        end_prefix_min_fraction=args.end_prefix_min_fraction,
        end_prefix_max_fraction=args.end_prefix_max_fraction,
        cache_dir=args.cache_dir,
        max_sequences=args.max_sequences,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_fasta and os.path.exists(args.val_fasta):
        val_dataset = ProteinDataset(
            fasta_file=args.val_fasta,
            tokenizer=tokenizer,
            l_max=args.l_max,
            mask_prob=args.mask_prob,
            objective=args.objective,
            prefix_length=args.prefix_length,
            prefix_min_fraction=args.prefix_min_fraction,
            prefix_max_fraction=args.prefix_max_fraction,
            end_prefix_prob=args.end_prefix_prob,
            end_prefix_min_fraction=args.end_prefix_min_fraction,
            end_prefix_max_fraction=args.end_prefix_max_fraction,
            cache_dir=args.cache_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    if args.model_size:
        model = S4ProteinModel.from_preset(
            args.model_size,
            vocab_size=tokenizer.vocab_size,
            dropout=args.dropout,
            kernel_type=args.kernel_type,
            bidirectional=bidirectional,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        model = S4ProteinModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            dropout=args.dropout,
            kernel_type=args.kernel_type,
            bidirectional=bidirectional,
            eos_token_id=tokenizer.eos_token_id,
        )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    if args.grad_accum_steps > 1:
        total_steps = (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_state = ckpt.get("model_state_dict", ckpt)
        checkpoint_vocab_size = raw_state["embed.weight"].shape[0]
        state = adapt_state_dict_vocab(raw_state, model.vocab_size)
        model.load_state_dict(state, strict=False)
        can_load_optimizer = checkpoint_vocab_size == model.vocab_size
        if "optimizer" in ckpt and can_load_optimizer:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("Skipping optimizer state because checkpoint tensor shapes changed.")
        elif "optimizer" in ckpt:
            print("Skipping optimizer state because checkpoint vocab size changed.")
        if "scheduler" in ckpt and can_load_optimizer:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except ValueError:
                print("Skipping scheduler state because checkpoint tensor shapes changed.")
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

    os.makedirs(args.output_dir, exist_ok=True)
    run = None
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, config=vars(args))
        except Exception:
            pass

    use_amp = args.use_amp and device.type == "cuda"
    best_path = os.path.join(args.output_dir, "best.pt")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args, global_step, use_amp
        )
        val_loss = validate(
            model,
            val_loader,
            device,
            args.objective,
            args.eos_loss_weight,
        ) if val_loader else train_loss

        if run:
            try:
                run.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})
            except Exception:
                pass

        save = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "objective": args.objective,
            "bidirectional": bidirectional,
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": model.d_model,
                "d_state": model.d_state,
                "n_layers": model.n_layers,
                "kernel_type": model.kernel_type,
                "eos_token_id": tokenizer.eos_token_id,
                "eos_loss_weight": args.eos_loss_weight,
                "end_prefix_prob": args.end_prefix_prob,
                "end_prefix_min_fraction": args.end_prefix_min_fraction,
                "end_prefix_max_fraction": args.end_prefix_max_fraction,
            },
        }
        torch.save(save, os.path.join(args.output_dir, "last.pt"))

        if args.save_best and (not os.path.exists(best_path) or val_loss < best_val_loss):
            best_val_loss = val_loss
            save["best_val_loss"] = best_val_loss
            torch.save(save, best_path)

        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}")

    if run:
        try:
            run.finish()
        except Exception:
            pass
    print("Training done. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
