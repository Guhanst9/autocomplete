import argparse
import csv
import json
from itertools import groupby
from pathlib import Path


def longest_run(sequence):
    return max((sum(1 for _ in group) for _, group in groupby(sequence)), default=0)


def summarize(path, bin_size=50):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 614:
        raise ValueError(f"expected 614 Rosa windows in {path}, found {len(rows)}")
    target_length = 512
    position_matches = [0] * target_length
    runs = []
    for row_number, row in enumerate(rows, start=2):
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        if len(generated) != target_length or len(truth) != target_length:
            raise ValueError(f"invalid sequence length in {path} row {row_number}")
        if set(generated) - set("ACGT"):
            raise ValueError(f"invalid generated base in {path} row {row_number}")
        for position, (left, right) in enumerate(zip(generated, truth)):
            position_matches[position] += left == right
        runs.append(longest_run(generated))

    position_accuracy = [100.0 * matches / len(rows) for matches in position_matches]
    bins = []
    for start in range(0, target_length, bin_size):
        values = position_accuracy[start : start + bin_size]
        bins.append(
            {
                "start": start + 1,
                "end": start + len(values),
                "accuracy_percent": sum(values) / len(values),
            }
        )
    return {
        "windows": len(rows),
        "accuracy_percent": sum(position_accuracy) / target_length,
        "first_100_accuracy_percent": sum(position_accuracy[:100]) / 100,
        "final_100_accuracy_percent": sum(position_accuracy[-100:]) / 100,
        "longest_run": max(runs),
        "rows_with_runs_over_20": sum(run > 20 for run in runs),
        "distance_bins": bins,
    }


def compare(control, experiment):
    deltas = {
        key: experiment[key] - control[key]
        for key in (
            "accuracy_percent",
            "first_100_accuracy_percent",
            "final_100_accuracy_percent",
        )
    }
    checks = {
        "final_100_improved_by_1pp": deltas["final_100_accuracy_percent"] >= 1.0,
        "overall_regression_within_0_5pp": deltas["accuracy_percent"] >= -0.5,
        "no_runs_over_20": experiment["rows_with_runs_over_20"] == 0,
    }
    return {
        "control": control,
        "experiment": experiment,
        "deltas": deltas,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare matched control and recursive recovery pilots."
    )
    parser.add_argument("--control-csv", required=True)
    parser.add_argument("--experiment-csv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(summarize(args.control_csv), summarize(args.experiment_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Pilot pass: {'yes' if result['pass'] else 'no'}")
    print(json.dumps(result["deltas"], indent=2))


if __name__ == "__main__":
    main()
