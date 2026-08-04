import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sliding_manifests(output_dir: str) -> list[Path]:
    return sorted(Path(output_dir).glob("sliding*/sliding_manifest.csv"))


def generated_manifest_rows(output_dir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest in sliding_manifests(output_dir):
        for row in read_csv(manifest):
            csv_path = Path(row["csv_path"])
            if row["status"] in {"generated", "skipped-existing"} and csv_path.exists():
                row = dict(row)
                row["manifest"] = str(manifest)
                rows.append(row)
    return rows


def load_window_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        for row in read_csv(Path(manifest_row["csv_path"])):
            merged = dict(row)
            merged["group"] = manifest_row["group"]
            merged["exposure"] = manifest_row["exposure"]
            merged["seed"] = manifest_row["seed"]
            merged["source_csv"] = manifest_row["csv_path"]
            all_rows.append(merged)
    return all_rows


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        accuracies = [float(row["accuracy_percent"]) for row in group_rows]
        gc_diffs = [float(row["gc_difference_percent"]) for row in group_rows]
        longest_runs = [int(row["longest_generated_run"]) for row in group_rows]
        n_counts = [int(row["n_count"]) for row in group_rows]
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(
            {
                "rows": len(group_rows),
                "avg_accuracy_percent": f"{mean(accuracies):.2f}",
                "min_accuracy_percent": f"{min(accuracies):.2f}",
                "max_accuracy_percent": f"{max(accuracies):.2f}",
                "avg_gc_difference_percent": f"{mean(gc_diffs):.2f}",
                "max_longest_generated_run": max(longest_runs),
                "runs_over_20": sum(run > 20 for run in longest_runs),
                "n_count": sum(n_counts),
            }
        )
        output.append(item)
    return output


def run_summarize(output_dir: str, reports_dir: str) -> None:
    manifest_rows = generated_manifest_rows(output_dir)
    if not manifest_rows:
        raise ValueError(f"No sliding CSVs found under {output_dir}")
    window_rows = load_window_rows(manifest_rows)
    if not window_rows:
        raise ValueError("No window rows found in sliding CSVs")

    reports = Path(reports_dir)
    region_rows = aggregate(
        window_rows,
        ["accession", "group", "exposure", "decoding_mode", "seed", "region"],
    )
    repeat_rows = aggregate(
        window_rows,
        ["accession", "group", "exposure", "decoding_mode", "seed"],
    )
    decoding_rows = aggregate(
        window_rows,
        ["decoding_mode", "seed"],
    )

    write_csv(
        reports / "region_accuracy.csv",
        [
            "accession",
            "group",
            "exposure",
            "decoding_mode",
            "seed",
            "region",
            "rows",
            "avg_accuracy_percent",
            "min_accuracy_percent",
            "max_accuracy_percent",
            "avg_gc_difference_percent",
            "max_longest_generated_run",
            "runs_over_20",
            "n_count",
        ],
        region_rows,
    )
    write_csv(
        reports / "repeat_gc_invalid_summary.csv",
        [
            "accession",
            "group",
            "exposure",
            "decoding_mode",
            "seed",
            "rows",
            "avg_accuracy_percent",
            "min_accuracy_percent",
            "max_accuracy_percent",
            "avg_gc_difference_percent",
            "max_longest_generated_run",
            "runs_over_20",
            "n_count",
        ],
        repeat_rows,
    )
    write_csv(
        reports / "decoding_comparison.csv",
        [
            "decoding_mode",
            "seed",
            "rows",
            "avg_accuracy_percent",
            "min_accuracy_percent",
            "max_accuracy_percent",
            "avg_gc_difference_percent",
            "max_longest_generated_run",
            "runs_over_20",
            "n_count",
        ],
        decoding_rows,
    )

    print("Summary stage complete")
    print(f"  Sliding CSVs: {len(manifest_rows)}")
    print(f"  Window rows: {len(window_rows)}")
    print(f"  Reports: {reports}")

