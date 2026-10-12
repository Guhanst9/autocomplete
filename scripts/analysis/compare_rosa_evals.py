import argparse
import csv
import json
from pathlib import Path


def longest_run(sequence: str) -> int:
    best = current = 0
    previous = None
    for base in sequence:
        if base == previous:
            current += 1
        else:
            previous = base
            current = 1
        best = max(best, current)
    return best


def summarize(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 614:
        raise ValueError(f"{path} contains {len(rows)} rows, expected 614")

    total_matches = first_matches = last_matches = 0
    row_runs = []
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        lengths = (len(prompt), len(generated), len(truth))
        if lengths != (512, 512, 512):
            raise ValueError(f"{path} row {index} has prompt/generated/true lengths {lengths}")
        if row["decoding_mode"] != "sampled":
            raise ValueError(f"{path} row {index} is not sampled decoding")
        total_matches += sum(a == b for a, b in zip(generated, truth))
        first_matches += sum(a == b for a, b in zip(generated[:100], truth[:100]))
        last_matches += sum(a == b for a, b in zip(generated[-100:], truth[-100:]))
        row_runs.append(longest_run(generated))

    return {
        "csv": str(path),
        "rows": len(rows),
        "prompt_length": 512,
        "generate_length": 512,
        "decoding_mode": "sampled",
        "recursive_per_base_accuracy": total_matches / (512 * len(rows)),
        "first_100_accuracy": first_matches / (100 * len(rows)),
        "last_100_accuracy": last_matches / (100 * len(rows)),
        "longest_repeated_base_run": max(row_runs),
        "rows_with_runs_over_20": sum(run > 20 for run in row_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched circular Rosa evaluations.")
    parser.add_argument("--triplet-csv", type=Path, required=True)
    parser.add_argument("--one-base-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    triplet = summarize(args.triplet_csv)
    one_base = summarize(args.one_base_csv)
    comparison = {
        "triplet": triplet,
        "one_base": one_base,
        "triplet_minus_one_base": {
            key: triplet[key] - one_base[key]
            for key in (
                "recursive_per_base_accuracy",
                "first_100_accuracy",
                "last_100_accuracy",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
