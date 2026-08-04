import csv
import json
from pathlib import Path
from typing import Any

from src.biological_eval.config import require_keys


def read_rows(path: str) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {path}")
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Baseline CSV has no rows: {path}")
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float | int]:
    accuracies = [float(row["accuracy_percent"]) for row in rows]
    longest_runs = [int(row["longest_generated_run"]) for row in rows]
    n_count = sum(int(row["n_count"]) for row in rows)
    return {
        "rows": len(rows),
        "accuracy_percent": round(sum(accuracies) / len(accuracies), 2),
        "runs_over_20": sum(run > 20 for run in longest_runs),
        "max_run": max(longest_runs),
        "n_count": n_count,
    }


def assert_metric(name: str, observed: float | int, expected: float | int) -> None:
    if observed != expected:
        raise ValueError(f"{name} drifted: observed {observed}, expected {expected}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_baseline_check(config: dict[str, Any], output_dir: str) -> None:
    require_keys(config, ["baseline_csvs", "known_rosa_baseline"])
    paths = config["baseline_csvs"]
    expected = config["known_rosa_baseline"]

    greedy = summarize_rows(read_rows(paths["greedy"]))
    sampled = summarize_rows(read_rows(paths["sampled_t08_seed13"]))

    assert_metric("greedy rows", greedy["rows"], 614)
    assert_metric("greedy accuracy", greedy["accuracy_percent"], expected["greedy_accuracy_percent"])
    assert_metric("greedy runs over 20", greedy["runs_over_20"], expected["greedy_runs_over_20"])
    assert_metric("greedy max run", greedy["max_run"], expected["greedy_max_run"])

    assert_metric("sampled t0.8 rows", sampled["rows"], 614)
    assert_metric(
        "sampled t0.8 accuracy",
        sampled["accuracy_percent"],
        expected["sampled_t08_accuracy_percent"],
    )
    assert_metric(
        "sampled t0.8 runs over 20",
        sampled["runs_over_20"],
        expected["sampled_t08_runs_over_20"],
    )
    assert_metric("sampled t0.8 max run", sampled["max_run"], expected["sampled_t08_max_run"])

    output_path = Path(output_dir) / "baseline_check.json"
    write_json(
        output_path,
        {
            "greedy": greedy,
            "sampled_t08_seed13": sampled,
            "status": "passed",
        },
    )
    print("Baseline check passed")
    print(f"  Greedy accuracy: {greedy['accuracy_percent']:.2f}%")
    print(f"  Greedy runs >20: {greedy['runs_over_20']}/{greedy['rows']}")
    print(f"  Greedy max run: {greedy['max_run']}")
    print(f"  Sampled t0.8 accuracy: {sampled['accuracy_percent']:.2f}%")
    print(f"  Sampled t0.8 runs >20: {sampled['runs_over_20']}/{sampled['rows']}")
    print(f"  Sampled t0.8 max run: {sampled['max_run']}")
    print(f"  Output: {output_path}")

