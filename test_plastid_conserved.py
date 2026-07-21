import argparse
import math
from collections import defaultdict
from difflib import SequenceMatcher

import torch

from run_plastid import PlastidTokenizer, get_device, stream_fasta, tokenizer_from_checkpoint
from src.models.s4_model import S4ProteinModel


def parse_args():
    parser = argparse.ArgumentParser(description="Test plastid checkpoint on a conserved DNA window.")
    parser.add_argument("--checkpoint", type=str, default="outputs/plastid_7m/best.pt")
    parser.add_argument("--fasta_file", type=str, default="data/plastid/plant_chloroplast_refseq_500.fna.gz")
    parser.add_argument("--max_records", type=int, default=50)
    parser.add_argument("--kmer_length", type=int, default=28)
    parser.add_argument("--prompt_length", type=int, default=160)
    parser.add_argument("--suffix_length", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    return parser.parse_args()


def load_records(path: str, tokenizer: PlastidTokenizer, max_records: int) -> list[tuple[str, str]]:
    records = []
    for header, sequence in stream_fasta(path):
        normalized = tokenizer.normalize(sequence)
        if normalized:
            records.append((header, normalized))
        if len(records) >= max_records:
            break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def find_conserved_kmer(records: list[tuple[str, str]], k: int) -> tuple[str, list[int]]:
    support: dict[str, set[int]] = defaultdict(set)
    for record_idx, (_, sequence) in enumerate(records):
        seen = set()
        for i in range(0, max(0, len(sequence) - k + 1), max(1, k // 2)):
            kmer = sequence[i : i + k]
            if "N" in kmer:
                continue
            seen.add(kmer)
        for kmer in seen:
            support[kmer].add(record_idx)
    if not support:
        raise ValueError("No conserved k-mers found. Try lowering --kmer_length.")

    best_kmer, best_records = max(
        support.items(),
        key=lambda item: (len(item[1]), item[0].count("G") + item[0].count("C")),
    )
    return best_kmer, sorted(best_records)


def extract_target_window(
    records: list[tuple[str, str]],
    kmer: str,
    record_indices: list[int],
    prompt_length: int,
    suffix_length: int,
) -> tuple[str, str, int]:
    total_len = prompt_length + suffix_length
    for record_idx in record_indices:
        header, sequence = records[record_idx]
        pos = sequence.find(kmer)
        if pos < 0:
            continue
        start = max(0, pos - prompt_length + len(kmer))
        if start + total_len <= len(sequence):
            return header, sequence[start : start + total_len], pos
        start = max(0, len(sequence) - total_len)
        if len(sequence[start : start + total_len]) == total_len:
            return header, sequence[start : start + total_len], pos
    raise ValueError("Could not extract a full target window around the conserved k-mer.")


def load_model(ckpt: dict, tokenizer: PlastidTokenizer, device: torch.device) -> S4ProteinModel:
    config = ckpt.get("model_config", {})
    model = S4ProteinModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 320),
        d_state=config.get("d_state", 96),
        n_layers=config.get("n_layers", 8),
        kernel_type=config.get("kernel_type", "diag"),
        bidirectional=False,
        model_variant=config.get("model_variant", "legacy"),
        l_max=config.get("l_max"),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=config.get("l_max", 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def teacher_forced_metrics(
    model: S4ProteinModel,
    tokenizer: PlastidTokenizer,
    target_window: str,
    prompt_length: int,
    device: torch.device,
) -> dict[str, float]:
    tokens = tokenizer.encode(target_window)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    target_ids = torch.tensor([tokens[1:] + [tokenizer.pad_token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[:, max(0, prompt_length - 1) : len(tokens) - 1] = 1

    logits = model(input_ids, attention_mask=attention_mask)
    loss = model.compute_loss(
        input_ids,
        target_ids,
        attention_mask,
        loss_mask=loss_mask,
        objective="autocomplete",
    )
    score_mask = loss_mask == 1
    scored_logits = logits[score_mask]
    scored_targets = target_ids[score_mask]
    top1 = (scored_logits.argmax(dim=-1) == scored_targets).float().mean().item()
    top3 = scored_logits.topk(min(3, scored_logits.shape[-1]), dim=-1).indices
    top3_acc = (top3 == scored_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {
        "loss": loss.item(),
        "perplexity": math.exp(loss.item()),
        "top1": top1,
        "top3": top3_acc,
    }


@torch.no_grad()
def generate(
    model: S4ProteinModel,
    tokenizer: PlastidTokenizer,
    prompt: str,
    args,
    device: torch.device,
) -> str:
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.suffix_length,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=True,
        forbidden_token_ids=(tokenizer.pad_token_id, tokenizer.unk_token_id),
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=min(20, args.suffix_length),
    )
    return tokenizer.decode(out[0].tolist(), stop_at_eos=True)


def positional_identity(a: str, b: str) -> float:
    overlap = min(len(a), len(b))
    if overlap == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:overlap], b[:overlap])) / overlap


def main():
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = tokenizer_from_checkpoint(ckpt)
    records = load_records(args.fasta_file, tokenizer, args.max_records)
    kmer, record_indices = find_conserved_kmer(records, args.kmer_length)
    header, target_window, position = extract_target_window(
        records,
        kmer,
        record_indices,
        args.prompt_length,
        args.suffix_length,
    )
    model = load_model(ckpt, tokenizer, device)

    prompt = target_window[: args.prompt_length]
    true_suffix = target_window[args.prompt_length :]
    completed = generate(model, tokenizer, prompt, args, device)
    generated_suffix = completed[len(prompt) :]
    teacher_forced = teacher_forced_metrics(model, tokenizer, target_window, args.prompt_length, device)

    print("Conserved plastid test")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Records searched: {len(records)}")
    print(f"  Conserved seed length: {args.kmer_length}")
    print(f"  Conserved seed support: {len(record_indices)}/{len(records)} records")
    print(f"  Target record: {header}")
    print(f"  Seed position in target: {position}")
    print()
    print("Teacher-forced target-window metrics")
    print(f"  top1 next-base accuracy: {teacher_forced['top1']:.4f}")
    print(f"  top3 next-base accuracy: {teacher_forced['top3']:.4f}")
    print(f"  perplexity: {teacher_forced['perplexity']:.4f}")
    print()
    print("Free generation similarity")
    print(f"  generated suffix length: {len(generated_suffix)}")
    print(f"  exact suffix position identity: {positional_identity(generated_suffix, true_suffix):.4f}")
    print(f"  suffix sequence similarity: {SequenceMatcher(None, generated_suffix, true_suffix).ratio():.4f}")
    print()
    print("Prompt:", prompt)
    print("True suffix:", true_suffix)
    print("Generated suffix:", generated_suffix)


if __name__ == "__main__":
    main()
