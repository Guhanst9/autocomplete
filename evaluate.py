"""
Evaluation: perplexity, masked accuracy, per-position and per-amino-acid breakdown.
"""
import argparse
import os
from collections import defaultdict

import torch
from tqdm import tqdm

try:
    from src.models.s4_model import S4ProteinModel
    from src.dataloaders.protein import ProteinDataset, ProteinTokenizer
except ImportError:
    from models.s4_model import S4ProteinModel
    from dataloaders.protein import ProteinDataset, ProteinTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fasta_file", type=str, default="data/protein/uniref50.fasta.gz")
    p.add_argument("--l_max", type=int, default=1024)
    p.add_argument("--mask_prob", type=float, default=0.15)
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


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_masked = 0
    correct_masked = 0
    correct_all = 0
    total_all = 0
    per_pos_correct = defaultdict(int)
    per_pos_total = defaultdict(int)
    per_aa_correct = defaultdict(int)
    per_aa_total = defaultdict(int)

    it = enumerate(dataloader)
    if max_batches:
        from itertools import islice
        it = islice(it, max_batches)

    for _, (input_ids, target_ids, attention_mask) in tqdm(it, total=max_batches, desc="Eval"):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = model.compute_loss(input_ids, target_ids, attention_mask)
        total_loss += loss.item()

        mask_token_id = getattr(model, "mask_token_id", 1)
        mask = (attention_mask == 1) & (input_ids == mask_token_id)
        pred = logits.argmax(dim=-1)

        n_masked = mask.sum().item()
        if n_masked > 0:
            correct_masked += (pred[mask] == target_ids[mask]).sum().item()
        total_masked += n_masked

        valid = attention_mask == 1
        total_all += valid.sum().item()
        correct_all += (pred[valid] == target_ids[valid]).sum().item()

        for b in range(input_ids.shape[0]):
            for s in range(attention_mask.shape[1]):
                if attention_mask[b, s].item() == 0:
                    continue
                pos = s
                aa = target_ids[b, s].item()
                per_pos_total[pos] += 1
                per_aa_total[aa] += 1
                if pred[b, s].item() == aa:
                    per_pos_correct[pos] += 1
                    per_aa_correct[aa] += 1

    n_batches = min(len(dataloader), max_batches or len(dataloader))
    n_batches = max(1, n_batches)
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    masked_acc = correct_masked / total_masked if total_masked else 0.0
    overall_acc = correct_all / total_all if total_all else 0.0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "masked_accuracy": masked_acc,
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = state["embed.weight"].shape[0]
    d_model = state["embed.weight"].shape[1]
    block_keys = [k for k in state if k.startswith("blocks.")]
    if block_keys:
        indices = [int(k.split(".")[1]) for k in block_keys if len(k.split(".")) >= 2 and k.split(".")[1].isdigit()]
        n_layers = max(indices) + 1 if indices else 6
    else:
        n_layers = 6
    n_layers = max(1, n_layers)

    model = S4ProteinModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
    )
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    tokenizer = ProteinTokenizer()
    
    print(f"📊 Evaluating checkpoint: {args.checkpoint}")
    print(f"   Device: {device}")
    print(f"   Model: d_model={d_model}, n_layers={n_layers}")
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
        cache_dir=args.cache_dir,
        max_sequences=args.max_sequences,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = evaluate_model(model, loader, device, args.max_batches)

    print("Loss:", results["loss"])
    print("Perplexity:", results["perplexity"])
    print("Masked token accuracy:", results["masked_accuracy"])
    print("Per-position accuracy (overall):", results["overall_accuracy"])

    if results["per_aa_total"]:
        print("\nPer-amino-acid accuracy (sample):")
        for aa, total in sorted(results["per_aa_total"].items(), key=lambda x: -x[1])[:10]:
            acc = results["per_aa_correct"].get(aa, 0) / total if total else 0
            name = tokenizer.idx_to_token.get(aa, str(aa))
            print(f"  {name} (id={aa}): {acc:.4f} ({results['per_aa_correct'].get(aa, 0)}/{total})")


if __name__ == "__main__":
    main()
