import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAINING_HISTORY = [
    (0, 1.3890, 0.8387, 0.6376, 0.9291, 0.4463, 78, 0.5312),
    (1, 1.0661, 0.7433, 0.6847, 0.9393, 0.4746, 100, 0.2500),
    (2, 0.9608, 0.6904, 0.7111, 0.9451, 0.4846, 18, 0.0000),
    (3, 0.8969, 0.6533, 0.7286, 0.9502, 0.4895, 79, 0.2188),
    (4, 0.8488, 0.6278, 0.7432, 0.9533, 0.5010, 65, 0.1250),
    (5, 0.8109, 0.6017, 0.7540, 0.9562, 0.5127, 113, 0.2500),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_training_history(reports_dir: str) -> Path:
    rows = [
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_top1": top1,
            "val_top3": top3,
            "free_acc": free_acc,
            "free_longest_run": longest_run,
            "free_runs_gt20": runs_gt20,
        }
        for epoch, train_loss, val_loss, top1, top3, free_acc, longest_run, runs_gt20 in TRAINING_HISTORY
    ]
    path = Path(reports_dir) / "training_history.csv"
    write_csv(path, list(rows[0]), rows)
    return path


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#2f6f73")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_line(path: Path, x_values: list[int], series: list[tuple[str, list[float]]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for label, values in series:
        plt.plot(x_values, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_figures(reports_dir: str) -> None:
    reports = Path(reports_dir)
    figure_dir = reports / "figures"
    history_path = write_training_history(reports_dir)
    history = read_csv(history_path)
    epochs = [int(row["epoch"]) for row in history]
    save_line(
        figure_dir / "training_validation_loss.png",
        epochs,
        [
            ("train loss", [float(row["train_loss"]) for row in history]),
            ("val loss", [float(row["val_loss"]) for row in history]),
        ],
        "training and validation loss",
        "loss",
    )
    save_line(
        figure_dir / "validation_accuracy.png",
        epochs,
        [
            ("top1", [100 * float(row["val_top1"]) for row in history]),
            ("top3", [100 * float(row["val_top3"]) for row in history]),
        ],
        "validation accuracy",
        "percent",
    )

    region = read_csv(reports / "region_accuracy.csv")
    if region:
        save_bar(
            figure_dir / "region_accuracy.png",
            [row["region"] for row in region],
            [float(row["avg_accuracy_percent"]) for row in region],
            "recursive accuracy by region",
            "accuracy (%)",
        )

    feature = read_csv(reports / "feature_summary.csv")
    if feature:
        save_bar(
            figure_dir / "feature_accuracy.png",
            [row["feature_label"] for row in feature],
            [float(row["avg_accuracy_percent"]) for row in feature],
            "recursive accuracy by feature",
            "accuracy (%)",
        )

    ir = read_csv(reports / "ir_summary.csv")
    if ir:
        save_bar(
            figure_dir / "ir_identity.png",
            [row["accession"] for row in ir],
            [float(row["avg_generated_ir_identity_percent"]) for row in ir],
            "generated inverted-repeat identity",
            "identity (%)",
        )

    print("Figure stage complete")
    print(f"  Training history: {history_path}")
    print(f"  Figures: {figure_dir}")


def table_preview(path: Path, max_rows: int = 5) -> str:
    rows = read_csv(path)
    if not rows:
        return "_not generated yet_"
    fieldnames = list(rows[0])
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row[key]) for key in fieldnames) + " |")
    return "\n".join(lines)


def run_report(config: dict[str, Any], reports_dir: str) -> None:
    reports = Path(reports_dir)
    report_path = Path("reports/plastid_s4d_evaluation.md")
    checkpoint = config["checkpoint"]
    content = f"""# Plastid S4D evaluation

## Motivation and question

This report freezes the current S4D plastid checkpoint and studies what it generates before changing training again. The main question is whether the model's generated DNA preserves local sequence identity, region behavior, annotated genes/RNAs, inverted repeats, and context-length effects.

## Dataset and cleaning

The model uses cleaned plastid FASTA data with high-N records removed and remaining N bases replaced before training. The frozen checkpoint is `{checkpoint}`. `NC_053550.1` is treated as a decoding-development genome because it was used while choosing sampled decoding at temperature `0.8`.

## Model setup

The checkpoint is the 16.57M-parameter S4D-v2 model trained with 1,024-base context, next-base prediction, two-pass recovery, and a weak homopolymer-ending loss. The primary generation mode for evaluation is sampled decoding at temperature `0.8`; greedy decoding is kept as a stability diagnostic.

## Decoding comparison

{table_preview(reports / "decoding_comparison.csv")}

## Genome-region results

{table_preview(reports / "region_accuracy.csv")}

## CDS, tRNA, and rRNA results

{table_preview(reports / "feature_summary.csv")}

## Inverted-repeat results

{table_preview(reports / "ir_summary.csv")}

## Context-length and top-k results

{table_preview(reports / "topk_summary.csv")}

Context-length outputs are stored in ignored CSV files under `outputs/plastid_biological_eval/` because they include generated sequences.

## Synthetic controls

Synthetic evaluation remains a separate check through `run_synthetic_eval.py`. It is useful for debugging repeated-base behavior without changing the biological evaluation panel.

## Limitations and next experiments

The current report is based on local panel smoke outputs unless the full stages are rerun. GenBank annotations are used for feature labels, while IRA/IRB currently falls back to sequence inference when annotation boundaries are unavailable. External NCBI test genomes and model-size comparisons should happen only after the local panel evaluation is reviewed.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content)
    print("Report stage complete")
    print(f"  Report: {report_path}")
