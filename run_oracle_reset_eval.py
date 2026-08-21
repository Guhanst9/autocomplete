import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import torch
from tqdm import tqdm

from run_decoding_sweep import evenly_spaced_starts
from src.dna.checkpoint import load_model
from src.dna.generation import generate_bases


def load_rows(path: Path, max_windows: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    if max_windows is None or max_windows >= len(rows):
        return rows
    starts = [int(row["window_start"]) for row in rows]
    selected = set(evenly_spaced_starts(starts, max_windows))
    return [row for row in rows if int(row["window_start"]) in selected]


def validate_rows(rows: list[dict[str, str]]) -> int:
    lengths = {len(row["true_suffix"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError("all true suffixes must have the same length")
    target_length = lengths.pop()
    for index, row in enumerate(rows, start=2):
        if not row["prompt"] or set(row["prompt"] + row["true_suffix"]) - set("ACGT"):
            raise ValueError(f"invalid DNA sequence in row {index}")
    return target_length


def longest_run(sequence: str) -> int:
    return max((sum(1 for _ in group) for _, group in groupby(sequence)), default=0)


def accuracy(generated: str, truth: str) -> float:
    return 100.0 * sum(a == b for a, b in zip(generated, truth)) / len(truth)


def distance_accuracy(generated: str, truth: str, width: int = 100) -> tuple[float, float]:
    length = min(width, len(truth))
    return accuracy(generated[:length], truth[:length]), accuracy(
        generated[-length:], truth[-length:]
    )


@torch.no_grad()
def generate_with_resets(
    model,
    tokenizer,
    device,
    rows: list[dict[str, str]],
    reset_interval: int,
    batch_size: int,
    temperature: float,
    seed: int,
) -> list[str]:
    if reset_interval <= 0:
        raise ValueError("reset interval must be positive")
    target_length = len(rows[0]["true_suffix"])
    generated = [""] * len(rows)
    torch.manual_seed(seed)

    offsets = range(0, target_length, reset_interval)
    total_batches = len(range(0, len(rows), batch_size)) * len(offsets)
    progress = tqdm(total=total_batches, desc=f"Reset {reset_interval}")
    for offset in offsets:
        chunk_length = min(reset_interval, target_length - offset)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            contexts = [row["prompt"] + row["true_suffix"][:offset] for row in batch]
            context_ids = [tokenizer.encode(context) for context in contexts]
            context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
            output = generate_bases(
                model,
                tokenizer,
                context_tensor,
                max_new_bases=chunk_length,
                sampling_temperature=temperature,
            )
            for local_index, token_ids in enumerate(output.tolist()):
                decoded = tokenizer.decode(token_ids, stop_at_eos=False)
                chunk = decoded[len(contexts[local_index]) :]
                generated[start + local_index] += chunk[:chunk_length]
            progress.update(1)
    progress.close()
    return generated


def summarize(rows: list[dict[str, str]], generated: list[str], reset_interval: int) -> dict:
    total_matches = total_bases = first_matches = last_matches = 0
    runs = []
    for row, sequence in zip(rows, generated):
        truth = row["true_suffix"]
        total_matches += sum(a == b for a, b in zip(sequence, truth))
        total_bases += len(truth)
        first_matches += sum(a == b for a, b in zip(sequence[:100], truth[:100]))
        last_matches += sum(a == b for a, b in zip(sequence[-100:], truth[-100:]))
        runs.append(longest_run(sequence))
    return {
        "reset_interval": reset_interval,
        "windows": len(rows),
        "accuracy_percent": 100.0 * total_matches / total_bases,
        "first_100_accuracy_percent": 100.0 * first_matches / (100 * len(rows)),
        "last_100_accuracy_percent": 100.0 * last_matches / (100 * len(rows)),
        "longest_run": max(runs),
        "rows_with_runs_over_20": sum(run > 20 for run in runs),
    }


def write_results(
    path: Path,
    rows: list[dict[str, str]],
    generated_by_interval: dict[str | int, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_start",
        "reset_interval",
        "accuracy_percent",
        "first_100_accuracy_percent",
        "last_100_accuracy_percent",
        "longest_run",
        "generated_suffix",
        "true_suffix",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for interval, generated_rows in generated_by_interval.items():
            for row, generated in zip(rows, generated_rows):
                first, last = distance_accuracy(generated, row["true_suffix"])
                writer.writerow(
                    {
                        "window_start": row["window_start"],
                        "reset_interval": interval,
                        "accuracy_percent": accuracy(generated, row["true_suffix"]),
                        "first_100_accuracy_percent": first,
                        "last_100_accuracy_percent": last,
                        "longest_run": longest_run(generated),
                        "generated_suffix": generated,
                        "true_suffix": row["true_suffix"],
                    }
                )


def parse_intervals(value: str) -> list[int]:
    intervals = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not intervals or any(interval <= 0 for interval in intervals):
        raise ValueError("reset intervals must contain positive integers")
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate periodic resets to true DNA history.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-intervals", default="25,50,100")
    parser.add_argument("--max-windows", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    rows = load_rows(args.baseline_csv, args.max_windows)
    target_length = validate_rows(rows)
    intervals = parse_intervals(args.reset_intervals)
    print(f"Using {len(rows)} evenly spaced windows with {target_length}-base targets")

    model, tokenizer, device = load_model(args.checkpoint)
    baseline_generated = [row["generated_suffix"] for row in rows]
    if any(len(sequence) != target_length for sequence in baseline_generated):
        raise ValueError("baseline CSV contains an incorrect generated length")
    generated_by_interval = {"none": baseline_generated}
    baseline_summary = summarize(rows, baseline_generated, reset_interval=0)
    baseline_summary["reset_interval"] = "none"
    summaries = [baseline_summary]
    print(
        f"reset=none accuracy={baseline_summary['accuracy_percent']:.2f}% "
        f"first100={baseline_summary['first_100_accuracy_percent']:.2f}% "
        f"last100={baseline_summary['last_100_accuracy_percent']:.2f}%"
    )
    for interval in intervals:
        generated = generate_with_resets(
            model,
            tokenizer,
            device,
            rows,
            interval,
            args.batch_size,
            args.temperature,
            args.seed,
        )
        if any(len(sequence) != target_length for sequence in generated):
            raise ValueError(f"reset {interval} produced an incorrect output length")
        generated_by_interval[interval] = generated
        summary = summarize(rows, generated, interval)
        summaries.append(summary)
        print(
            f"reset={interval} accuracy={summary['accuracy_percent']:.2f}% "
            f"first100={summary['first_100_accuracy_percent']:.2f}% "
            f"last100={summary['last_100_accuracy_percent']:.2f}%"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir / "oracle_reset_windows.csv", rows, generated_by_interval)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
