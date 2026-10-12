import argparse
import csv
import json
from pathlib import Path


def calculate_distance_accuracy(rows: list[dict], bin_size: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        raise ValueError("evaluation CSV contains no rows")
    lengths = {len(row["true_suffix"]) for row in rows}
    generated_lengths = {len(row["generated_suffix"]) for row in rows}
    if len(lengths) != 1 or lengths != generated_lengths:
        raise ValueError("all generated and true suffixes must have the same length")

    sequence_length = lengths.pop()
    matches = [0] * sequence_length
    for row in rows:
        for position, (generated, truth) in enumerate(
            zip(row["generated_suffix"], row["true_suffix"])
        ):
            matches[position] += int(generated == truth)

    position_rows = [
        {
            "position": position + 1,
            "accuracy_percent": 100 * count / len(rows),
        }
        for position, count in enumerate(matches)
    ]
    bin_rows = []
    for start in range(0, sequence_length, bin_size):
        end = min(sequence_length, start + bin_size)
        bin_rows.append(
            {
                "start_position": start + 1,
                "end_position": end,
                "bases": end - start,
                "accuracy_percent": 100 * sum(matches[start:end]) / (len(rows) * (end - start)),
            }
        )
    return position_rows, bin_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure generation accuracy by output distance.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=50)
    args = parser.parse_args()
    if args.bin_size <= 0:
        raise ValueError("bin size must be positive")

    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    position_rows, bin_rows = calculate_distance_accuracy(rows, args.bin_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "position_accuracy.csv", position_rows)
    write_csv(args.output_dir / "distance_bins.csv", bin_rows)
    summary = {
        "source_csv": str(args.csv),
        "windows": len(rows),
        "generated_length": len(position_rows),
        "bin_size": args.bin_size,
        "first_100_accuracy_percent": sum(
            row["accuracy_percent"] for row in position_rows[:100]
        )
        / 100,
        "final_100_accuracy_percent": sum(
            row["accuracy_percent"] for row in position_rows[-100:]
        )
        / 100,
        "distance_bins": bin_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in bin_rows:
        print(
            f"{row['start_position']:>3}-{row['end_position']:<3}: "
            f"{row['accuracy_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
