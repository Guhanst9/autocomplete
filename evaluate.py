"""
Evaluation: perplexity, masked accuracy, per-position and per-amino-acid breakdown.
"""
import argparse
import os
from collections import defaultdict

import torch
from tqdm import tqdm

try:
    from src.models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from src.dataloaders.protein import ProteinDataset, ProteinTokenizer
except ImportError:
    from models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from dataloaders.protein import ProteinDataset, ProteinTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fasta_file", type=str, default="data/protein/uniref50.fasta.gz")
    p.add_argument("--l_max", type=int, default=1024)
    p.add_argument("--objective", type=str, default="autocomplete", choices=["autocomplete", "masked"])
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--prefix_length", type=int, default=None)
    p.add_argument("--prefix_min_fraction", type=float, default=0.25)
    p.add_argument("--prefix_max_fraction", type=float, default=0.70)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Maximum number of batches to evaluate (for quick testing)")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Maximum number of sequences to load (for memory-limited systems). "
                        "E.g., --max_sequences 100000 for 100K sequences.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory to cache processed sequences for faster loading")
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


def infer_model_shape(state, ckpt):
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    d_model = state["embed.weight"].shape[1]
    d_state = model_config.get("d_state")
    kernel_type = model_config.get("kernel_type")

    if d_state is None:
        for key in ("blocks.0.s4_layer.kernel.log_A_real", "blocks.0.s4_layer.kernel.A"):
            if key in state:
                d_state = state[key].shape[1] * 2
                break
    if d_state is None:
        d_state = 64

    if kernel_type is None:
        kernel_type = "nplr" if "blocks.0.s4_layer.kernel.A" in state else "diag"

    block_keys = [k for k in state if k.startswith("blocks.")]
    if block_keys:
        indices = [int(k.split(".")[1]) for k in block_keys if len(k.split(".")) >= 2 and k.split(".")[1].isdigit()]
        n_layers = max(indices) + 1 if indices else 6
    else:
        n_layers = 6

    return d_model, d_state, max(1, n_layers), kernel_type


@torch.no_grad()
def evaluate_model(model, dataloader, device, objective="autocomplete", max_batches=None):
    model.eval()
    total_loss = 0.0
    total_scored = 0
    correct_topk = {1: 0, 3: 0, 5: 0}
    correct_all = 0
    total_all = 0
    eos_total = 0
    eos_correct = 0
    per_pos_correct = defaultdict(int)
    per_pos_total = defaultdict(int)
    per_aa_correct = defaultdict(int)
    per_aa_total = defaultdict(int)
    n_batches = 0

    it = enumerate(dataloader)
    if max_batches:
        from itertools import islice
        it = islice(it, max_batches)

    for _, batch in tqdm(it, total=max_batches, desc="Eval"):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = model.compute_loss(
            input_ids, target_ids, attention_mask,
            loss_mask=loss_mask, objective=objective,
        )
        total_loss += loss.item()
        n_batches += 1

        mask_token_id = getattr(model, "mask_token_id", 1)
        if objective == "masked":
            score_mask = (attention_mask == 1) & (input_ids == mask_token_id)
        else:
            score_mask = loss_mask == 1 if loss_mask is not None else (target_ids != model.pad_token_id)

        pred = logits.argmax(dim=-1)

        n_scored = score_mask.sum().item()
        if n_scored > 0:
            scored_logits = logits[score_mask]
            scored_targets = target_ids[score_mask]
            total_scored += n_scored
            for k in correct_topk:
                k_eff = min(k, logits.shape[-1])
                topk = scored_logits.topk(k_eff, dim=-1).indices
                correct_topk[k] += (topk == scored_targets.unsqueeze(-1)).any(dim=-1).sum().item()

            if getattr(model, "eos_token_id", None) is not None:
                eos_mask = scored_targets == model.eos_token_id
                eos_total += eos_mask.sum().item()
                if eos_mask.any():
                    eos_correct += (pred[score_mask][eos_mask] == model.eos_token_id).sum().item()

        valid = attention_mask == 1
        total_all += valid.sum().item()
        correct_all += (pred[valid] == target_ids[valid]).sum().item()

        for b in range(input_ids.shape[0]):
            for s in range(attention_mask.shape[1]):
                if not score_mask[b, s].item():
                    continue
                pos = s
                aa = target_ids[b, s].item()
                per_pos_total[pos] += 1
                per_aa_total[aa] += 1
                if pred[b, s].item() == aa:
                    per_pos_correct[pos] += 1
                    per_aa_correct[aa] += 1

    n_batches = max(1, n_batches)
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    topk_accuracy = {
        k: correct_topk[k] / total_scored if total_scored else 0.0
        for k in correct_topk
    }
    overall_acc = correct_all / total_all if total_all else 0.0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "topk_accuracy": topk_accuracy,
        "scored_token_accuracy": topk_accuracy[1],
        "total_scored_tokens": total_scored,
        "eos_accuracy": eos_correct / eos_total if eos_total else None,
        "eos_total": eos_total,
        "overall_accuracy": overall_acc,
        "per_pos_correct": dict(per_pos_correct),
        "per_pos_total": dict(per_pos_total),
        "per_aa_correct": dict(per_aa_correct),
        "per_aa_total": dict(per_aa_total),
    }


def get_device():
    """Get best available device: CUDA > MPS > CPU"""
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = tokenizer.vocab_size
    d_model, d_state, n_layers, kernel_type = infer_model_shape(state, ckpt)

    bidirectional = ckpt.get("bidirectional", args.objective == "masked") if isinstance(ckpt, dict) else args.objective == "masked"
    model = S4ProteinModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        kernel_type=kernel_type,
        bidirectional=bidirectional,
        eos_token_id=tokenizer.eos_token_id,
    )
    state = adapt_state_dict_vocab(state, model.vocab_size)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    
    print(f"📊 Evaluating checkpoint: {args.checkpoint}")
    print(f"   Device: {device}")
    print(f"   Model: d_model={d_model}, d_state={d_state}, n_layers={n_layers}, kernel_type={kernel_type}")
    print(f"   Objective: {args.objective}")
    print(f"   Bidirectional: {bidirectional}")
    if args.max_sequences:
        print(f"   Max sequences: {args.max_sequences:,}")
    if args.max_batches:
        print(f"   Max batches: {args.max_batches}")
    print()
    
    dataset = ProteinDataset(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        l_max=args.l_max,
        mask_prob=args.mask_prob,
        objective=args.objective,
        prefix_length=args.prefix_length,
        prefix_min_fraction=args.prefix_min_fraction,
        prefix_max_fraction=args.prefix_max_fraction,
        cache_dir=args.cache_dir,
        max_sequences=args.max_sequences,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = evaluate_model(model, loader, device, args.objective, args.max_batches)

    print("Loss:", results["loss"])
    print("Perplexity:", results["perplexity"])
    if args.objective == "masked":
        print("Masked token accuracy:", results["scored_token_accuracy"])
    else:
        print("Suffix token accuracy:", results["scored_token_accuracy"])
        print("Suffix top-3 accuracy:", results["topk_accuracy"][3])
        print("Suffix top-5 accuracy:", results["topk_accuracy"][5])
        if results["eos_accuracy"] is not None:
            print("EOS accuracy:", results["eos_accuracy"], f"({results['eos_total']} EOS targets)")
    print("Per-position accuracy (overall):", results["overall_accuracy"])
    print("Scored tokens:", results["total_scored_tokens"])

    if results["per_aa_total"]:
        print("\nPer-token accuracy (sample):")
        for aa, total in sorted(results["per_aa_total"].items(), key=lambda x: -x[1])[:10]:
            acc = results["per_aa_correct"].get(aa, 0) / total if total else 0
            name = tokenizer.idx_to_token.get(aa, str(aa))
            print(f"  {name} (id={aa}): {acc:.4f} ({results['per_aa_correct'].get(aa, 0)}/{total})")


if __name__ == "__main__":
    main()
