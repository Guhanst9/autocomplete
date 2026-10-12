import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dna.checkpoint import load_model
from src.dna.data import stream_fasta
from src.dna.generation import generate_bases
from src.dna.prediction import TripletCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one-base and overlapping triplet DNA checkpoints."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--triplet-checkpoint", required=True)
    parser.add_argument("--fasta-file", required=True)
    parser.add_argument("--accession", default="NC_053550.1")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generate-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decoding-mode", choices=("greedy", "sampled"), default="sampled")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_accession(path: str, accession: str) -> tuple[str, str]:
    for header, sequence in stream_fasta(path):
        if header.split()[0] == accession:
            cleaned = sequence.upper().replace("U", "T")
            if set(cleaned) - set("ACGTN"):
                raise ValueError(f"{accession} contains unsupported DNA symbols")
            return header, cleaned
    raise ValueError(f"accession {accession} was not found in {path}")


def circular_slice(sequence: str, start: int, length: int) -> str:
    if not sequence or length < 0:
        raise ValueError("circular sequence must be nonempty and length nonnegative")
    return "".join(sequence[(start + offset) % len(sequence)] for offset in range(length))


def make_windows(
    sequence: str,
    prompt_length: int,
    generate_length: int,
    stride: int,
    max_windows: int | None,
) -> list[dict]:
    windows = []
    for start in range(0, len(sequence), stride):
        windows.append(
            {
                "window_start": start,
                "prompt": circular_slice(sequence, start, prompt_length),
                "truth": circular_slice(sequence, start + prompt_length, generate_length),
            }
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def base_teacher_metrics(model, tokenizer, windows: list[dict], batch_size: int) -> dict:
    total_loss = 0.0
    total_bases = 0
    correct_bases = 0
    exact_triplets = 0
    total_triplets = 0
    for start in tqdm(range(0, len(windows), batch_size), desc="Base teacher forcing"):
        batch = windows[start : start + batch_size]
        sequences = [item["prompt"] + item["truth"] for item in batch]
        inputs = torch.tensor(
            [tokenizer.encode(sequence[:-1]) for sequence in sequences],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        targets = torch.tensor(
            [tokenizer.encode(sequence[1:]) for sequence in sequences],
            dtype=torch.long,
            device=inputs.device,
        )
        logits = model(inputs, attention_mask=torch.ones_like(inputs))
        prompt_length = len(batch[0]["prompt"])
        scored_logits = logits[:, prompt_length - 1 :, :]
        scored_targets = targets[:, prompt_length - 1 :]
        base_ids = torch.tensor(
            [tokenizer.vocab[base] for base in "ACGT"],
            device=inputs.device,
        )
        restricted = scored_logits.index_select(-1, base_ids)
        target_classes = (scored_targets.unsqueeze(-1) == base_ids).long().argmax(dim=-1)
        total_loss += F.cross_entropy(restricted.flatten(0, 1), target_classes.flatten(), reduction="sum").item()
        predictions = base_ids[restricted.argmax(dim=-1)]
        correct_bases += (predictions == scored_targets).sum().item()
        total_bases += scored_targets.numel()
        for row, truth_row in zip(predictions, scored_targets):
            for offset in range(max(0, truth_row.numel() - 2)):
                exact_triplets += int(torch.equal(row[offset : offset + 3], truth_row[offset : offset + 3]))
                total_triplets += 1
    average_loss = total_loss / max(1, total_bases)
    return {
        "teacher_loss": average_loss,
        "perplexity": math.exp(average_loss),
        "base_normalized_perplexity": math.exp(average_loss),
        "exact_triplet_accuracy": exact_triplets / max(1, total_triplets),
        "per_base_accuracy": correct_bases / max(1, total_bases),
        "teacher_bases": total_bases,
        "teacher_triplets": total_triplets,
    }


def triplet_teacher_metrics(model, tokenizer, windows: list[dict], batch_size: int) -> dict:
    codec = TripletCodec(model.output_tokens)
    base_table = codec.base_ids(tokenizer, next(model.parameters()).device)
    total_loss = 0.0
    total_triplets = 0
    correct_triplets = 0
    correct_bases = 0
    for start in tqdm(range(0, len(windows), batch_size), desc="Triplet teacher forcing"):
        batch = windows[start : start + batch_size]
        sequences = [item["prompt"] + item["truth"] for item in batch]
        inputs = torch.tensor(
            [tokenizer.encode(sequence) for sequence in sequences],
            dtype=torch.long,
            device=base_table.device,
        )
        logits = model(inputs, attention_mask=torch.ones_like(inputs))
        prompt_length = len(batch[0]["prompt"])
        target_length = len(batch[0]["truth"])
        scored_logits = logits[:, prompt_length - 1 : prompt_length + target_length - 3, :]
        targets = torch.tensor(
            [
                [codec.encode(item["truth"][offset : offset + 3]) for offset in range(target_length - 2)]
                for item in batch
            ],
            dtype=torch.long,
            device=inputs.device,
        )
        total_loss += F.cross_entropy(scored_logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
        predictions = scored_logits.argmax(dim=-1)
        correct_triplets += (predictions == targets).sum().item()
        correct_bases += (base_table[predictions] == base_table[targets]).sum().item()
        total_triplets += targets.numel()
    average_loss = total_loss / max(1, total_triplets)
    return {
        "teacher_loss": average_loss,
        "perplexity": math.exp(average_loss),
        "base_normalized_perplexity": math.exp(average_loss / 3),
        "exact_triplet_accuracy": correct_triplets / max(1, total_triplets),
        "per_base_accuracy": correct_bases / max(1, 3 * total_triplets),
        "teacher_bases": 3 * total_triplets,
        "teacher_triplets": total_triplets,
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str,
    expected_unit: str,
    windows: list[dict],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    model, tokenizer, device = load_model(checkpoint)
    if model.prediction_unit != expected_unit:
        raise ValueError(
            f"expected a {expected_unit} checkpoint, but {checkpoint} is {model.prediction_unit}"
        )
    if any("N" in item["prompt"] + item["truth"] for item in windows):
        raise ValueError("comparison windows must contain only A/C/G/T")
    teacher = (
        base_teacher_metrics(model, tokenizer, windows, args.batch_size)
        if expected_unit == "base"
        else triplet_teacher_metrics(model, tokenizer, windows, args.batch_size)
    )

    torch.manual_seed(args.seed)
    rows = []
    total_matches = 0
    total_bases = 0
    position_matches = [0] * args.generate_length
    position_totals = [0] * args.generate_length
    for start in tqdm(range(0, len(windows), args.batch_size), desc=f"{expected_unit} recursive"):
        batch = windows[start : start + args.batch_size]
        prompt_ids = torch.tensor(
            [tokenizer.encode(item["prompt"]) for item in batch],
            dtype=torch.long,
            device=device,
        )
        output = generate_bases(
            model,
            tokenizer,
            prompt_ids,
            args.generate_length,
            sampling_temperature=args.temperature if args.decoding_mode == "sampled" else None,
        )
        for item, output_ids in zip(batch, output.tolist()):
            generated = tokenizer.decode(output_ids[len(item["prompt"]) :], stop_at_eos=False)
            matches = [left == right for left, right in zip(generated, item["truth"])]
            for position, matched in enumerate(matches):
                position_matches[position] += int(matched)
                position_totals[position] += 1
            total_matches += sum(matches)
            total_bases += len(matches)
            rows.append(
                {
                    "prediction_unit": expected_unit,
                    "window_start": item["window_start"],
                    "accuracy_percent": 100.0 * sum(matches) / max(1, len(matches)),
                    "first_100_accuracy_percent": 100.0 * sum(matches[:100]) / max(1, len(matches[:100])),
                    "last_100_accuracy_percent": 100.0 * sum(matches[-100:]) / max(1, len(matches[-100:])),
                    "prompt": item["prompt"],
                    "generated_suffix": generated,
                    "true_suffix": item["truth"],
                }
            )
    teacher.update(
        {
            "checkpoint": checkpoint,
            "prediction_unit": expected_unit,
            "recursive_accuracy": total_matches / max(1, total_bases),
            "recursive_bases": total_bases,
            "position_accuracy": [
                position_matches[index] / max(1, position_totals[index])
                for index in range(args.generate_length)
            ],
        }
    )
    return teacher, rows


def main() -> None:
    args = parse_args()
    if min(args.prompt_length, args.generate_length, args.stride, args.batch_size) <= 0:
        raise ValueError("length, stride, and batch arguments must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    header, sequence = load_accession(args.fasta_file, args.accession)
    windows = make_windows(
        sequence,
        args.prompt_length,
        args.generate_length,
        args.stride,
        args.max_windows,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for checkpoint, unit in (
        (args.base_checkpoint, "base"),
        (args.triplet_checkpoint, "triplet"),
    ):
        summary, rows = evaluate_checkpoint(checkpoint, unit, windows, args)
        summaries.append(summary)
        all_rows.extend(rows)

    with (output_dir / "window_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    metadata = {
        "accession": args.accession,
        "header": header,
        "genome_length": len(sequence),
        "windows": len(windows),
        "prompt_length": args.prompt_length,
        "generate_length": args.generate_length,
        "stride": args.stride,
        "decoding_mode": args.decoding_mode,
        "temperature": args.temperature if args.decoding_mode == "sampled" else None,
        "seed": args.seed,
        "models": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prediction-unit comparison complete")
    for summary in summaries:
        print(
            f"  {summary['prediction_unit']}: "
            f"triplet={100 * summary['exact_triplet_accuracy']:.2f}% "
            f"per_base={100 * summary['per_base_accuracy']:.2f}% "
            f"base_ppl={summary['base_normalized_perplexity']:.3f} "
            f"recursive={100 * summary['recursive_accuracy']:.2f}%"
        )
    print(f"  Summary: {output_dir / 'summary.json'}")
    print(f"  Windows: {output_dir / 'window_comparison.csv'}")


if __name__ == "__main__":
    main()
