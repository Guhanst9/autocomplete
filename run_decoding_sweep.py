import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def longest_run(sequence: str) -> int:
    best = current = 0
    previous = ""
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        best = max(best, current)
    return best


def summarize_csv(path: Path, temperature: float, seed: int) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no evaluation rows found in {path}")

    matches = first_matches = last_matches = total_bases = 0
    first_bases = last_bases = 0
    runs = []
    for row_number, row in enumerate(rows, start=2):
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        if not generated or len(generated) != len(truth):
            raise ValueError(f"invalid generated/true lengths in {path} row {row_number}")
        if set(generated) - set("ACGT"):
            raise ValueError(f"invalid generated base in {path} row {row_number}")
        matches += sum(left == right for left, right in zip(generated, truth))
        total_bases += len(truth)
        first_length = min(100, len(truth))
        last_length = min(100, len(truth))
        first_matches += sum(
            left == right for left, right in zip(generated[:first_length], truth[:first_length])
        )
        last_matches += sum(
            left == right for left, right in zip(generated[-last_length:], truth[-last_length:])
        )
        first_bases += first_length
        last_bases += last_length
        runs.append(longest_run(generated))

    return {
        "temperature": temperature,
        "seed": seed,
        "rows": len(rows),
        "accuracy_percent": 100 * matches / total_bases,
        "first_100_accuracy_percent": 100 * first_matches / first_bases,
        "last_100_accuracy_percent": 100 * last_matches / last_bases,
        "longest_run": max(runs),
        "rows_with_runs_over_20": sum(run > 20 for run in runs),
        "csv": str(path),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for temperature in sorted({row["temperature"] for row in rows}):
        group = [row for row in rows if row["temperature"] == temperature]
        accuracies = [row["accuracy_percent"] for row in group]
        output.append(
            {
                "temperature": temperature,
                "seeds": len(group),
                "mean_accuracy_percent": statistics.mean(accuracies),
                "accuracy_std_percent": statistics.stdev(accuracies) if len(group) > 1 else 0.0,
                "mean_first_100_accuracy_percent": statistics.mean(
                    row["first_100_accuracy_percent"] for row in group
                ),
                "mean_last_100_accuracy_percent": statistics.mean(
                    row["last_100_accuracy_percent"] for row in group
                ),
                "maximum_run": max(row["longest_run"] for row in group),
                "total_rows_with_runs_over_20": sum(
                    row["rows_with_runs_over_20"] for row in group
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_numbers(value: str, number_type):
    return [number_type(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched sampled-decoding evaluations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accession", default="NC_053550.1")
    parser.add_argument("--temperatures", default="0.6,0.7,0.8")
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generate-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    temperatures = parse_numbers(args.temperatures, float)
    seeds = parse_numbers(args.seeds, int)
    if not temperatures or not seeds or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures and seeds must be non-empty, with positive temperatures")

    results = []
    for temperature in temperatures:
        temperature_name = str(temperature).replace(".", "p")
        for seed in seeds:
            run_dir = args.output_dir / f"temperature_{temperature_name}_seed_{seed}"
            csv_path = run_dir / f"{args.accession}_windows.csv"
            if not args.summarize_only and (args.overwrite or not csv_path.exists()):
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "run_sliding_eval.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--fasta_file",
                    args.fasta_file,
                    "--accession",
                    args.accession,
                    "--prompt_length",
                    str(args.prompt_length),
                    "--generate_length",
                    str(args.generate_length),
                    "--stride",
                    str(args.stride),
                    "--circular",
                    "--batch_size",
                    str(args.batch_size),
                    "--decoding_mode",
                    "sampled",
                    "--temperature",
                    str(temperature),
                    "--seed",
                    str(seed),
                    "--output_dir",
                    str(run_dir),
                ]
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            if not csv_path.exists():
                raise FileNotFoundError(f"missing sweep result: {csv_path}")
            result = summarize_csv(csv_path, temperature, seed)
            results.append(result)
            print(
                f"temperature={temperature:.2f} seed={seed} "
                f"accuracy={result['accuracy_percent']:.2f}% "
                f"max_run={result['longest_run']}"
            )

    write_csv(args.output_dir / "seed_results.csv", results)
    write_csv(args.output_dir / "temperature_summary.csv", aggregate(results))
    print(f"Summary: {args.output_dir / 'temperature_summary.csv'}")


if __name__ == "__main__":
    main()
