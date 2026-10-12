"""Create publication figures and a compact results draft from final-test CSVs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


NEURAL_ORDER = [
    "s4d-4.13m",
    "s4d-16.60m",
    "s4d-26.98m",
    "transformer-16.44m",
]
LABELS = {
    "s4d-4.13m": "S4D 4.13M",
    "s4d-16.60m": "S4D 16.60M",
    "s4d-26.98m": "S4D 26.98M",
    "transformer-16.44m": "Transformer 16.44M",
}
COLORS = {
    "s4d-4.13m": "#4477AA",
    "s4d-16.60m": "#228833",
    "s4d-26.98m": "#CC6677",
    "transformer-16.44m": "#AA3377",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_progress(figure_dir: Path) -> None:
    training_files = {
        "s4d-4.13m": Path("outputs/temp/loss_curves/s4d_4.13m_validation_loss.csv"),
        "s4d-16.60m": Path("outputs/temp/loss_curves/s4d_16.60m_validation_loss.csv"),
        "s4d-26.98m": Path("outputs/temp/loss_curves/s4d_26.98m_validation_loss.csv"),
        "transformer-16.44m": Path("outputs/temp/loss_curves/transformer_16.44m_validation_loss.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for model in NEURAL_ORDER:
        rows = read_csv(training_files[model])
        epochs = [int(row["epoch_index"]) + 1 for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        accuracies = [100.0 * float(row["per_base_accuracy"]) for row in rows]
        axes[0].plot(epochs, losses, linewidth=2, color=COLORS[model], label=LABELS[model])
        axes[1].plot(epochs, accuracies, linewidth=2, color=COLORS[model], label=LABELS[model])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation per-base accuracy (%)")
    axes[0].set_xlim(left=1)
    axes[1].set_xlim(left=1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, figure_dir, "training_validation_progress")


def longest_run(sequence: str) -> int:
    best = current = 0
    previous = ""
    for base in sequence:
        current = current + 1 if base == previous else 1
        best = max(best, current)
        previous = base
    return best


def plot_test_perplexity(root: Path, figure_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(root / "teacher_forced_metrics.csv")
        if row["model"] in NEURAL_ORDER
    ]
    values = {row["model"]: float(row["base_normalized_perplexity"]) for row in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y_positions = list(range(len(NEURAL_ORDER)))
    for y, model in zip(y_positions, NEURAL_ORDER):
        ax.scatter(values[model], y, color=COLORS[model], s=60)
        ax.text(values[model] + 0.012, y, f"{values[model]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y_positions, [LABELS[m] for m in NEURAL_ORDER])
    ax.set_xlabel("Base-normalized perplexity")
    ax.set_xlim(1.20, 1.50)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figure_dir, "untouched_test_perplexity")


def plot_test_perplexity_bar(root: Path, figure_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(root / "teacher_forced_metrics.csv")
        if row["model"] in NEURAL_ORDER
    ]
    values = {row["model"]: float(row["base_normalized_perplexity"]) for row in rows}
    models = list(reversed(NEURAL_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bars = ax.barh(
        range(len(models)),
        [values[model] for model in models],
        color=[COLORS[model] for model in models],
        height=0.62,
    )
    ax.set_yticks(range(len(models)), [LABELS[model] for model in models])
    ax.set_xlabel("Base-normalized perplexity")
    ax.set_xlim(0, 1.55)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    for bar, model in zip(bars, models):
        ax.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{values[model]:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    save_figure(fig, figure_dir, "untouched_test_perplexity_bar")


def plot_alignment_identity(root: Path, figure_dir: Path) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(root / "model_comparison.csv"):
        if row["model"] in NEURAL_ORDER:
            grouped[row["model"]].append(
                (
                    float(row["accuracy_percent"]),
                    float(row["mean_global_alignment_identity_percent"]),
                )
            )
    values = {
        model: (
            mean([pair[0] for pair in grouped[model]]),
            mean([pair[1] for pair in grouped[model]]),
        )
        for model in NEURAL_ORDER
    }
    models = list(reversed(NEURAL_ORDER))
    y_positions = list(range(len(models)))
    bar_height = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    exact_values = [values[model][0] for model in models]
    aligned_values = [values[model][1] for model in models]
    ax.barh(
        [y - bar_height / 2 for y in y_positions],
        exact_values,
        height=bar_height,
        color="#777777",
        label="exact position",
    )
    ax.barh(
        [y + bar_height / 2 for y in y_positions],
        aligned_values,
        height=bar_height,
        color=[COLORS[model] for model in models],
        label="global alignment",
    )
    for y, exact, aligned in zip(y_positions, exact_values, aligned_values):
        ax.text(exact + 0.7, y - bar_height / 2, f"{exact:.1f}", va="center", fontsize=8)
        ax.text(aligned + 0.7, y + bar_height / 2, f"{aligned:.1f}", va="center", fontsize=8)
    ax.set_yticks(y_positions, [LABELS[model] for model in models])
    ax.set_xlabel("Accuracy or identity (%)")
    ax.set_xlim(50, 90)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.26, top=0.98)
    save_figure(fig, figure_dir, "exact_accuracy_vs_alignment_identity")


def plot_repeat_behavior(root: Path, figure_dir: Path) -> None:
    runs_by_model: dict[str, list[int]] = {}
    for model in NEURAL_ORDER:
        model_runs = []
        for seed_dir in (root / "models" / model).glob("seed_*"):
            for path in seed_dir.glob("*_windows.csv"):
                model_runs.extend(
                    longest_run(row["generated_suffix"])
                    for row in read_csv(path)
                )
        runs_by_model[model] = model_runs

    target_rows = []
    for path in (root / "models" / "s4d-26.98m" / "seed_13").glob("*_windows.csv"):
        target_rows.extend(read_csv(path))
    target_runs = [longest_run(row["true_suffix"]) for row in target_rows]

    plotted_models = ["s4d-16.60m", "s4d-26.98m", "transformer-16.44m"]
    all_values = [runs_by_model[model] for model in plotted_models] + [target_runs]
    values = [[run for run in runs if run >= 20] for runs in all_values]
    labels = ["16.60M S4D", "26.98M S4D", "Transformer", "True suffixes"]
    colors = [COLORS[model] for model in plotted_models] + ["#777777"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for index, (runs, color) in enumerate(zip(values, colors), start=1):
        if not runs:
            ax.text(index, 22, "none", ha="center", va="center", fontsize=8)
            continue
        ordered = sorted(runs)
        offsets = [0.0] if len(ordered) == 1 else [
            -0.18 + 0.36 * position / (len(ordered) - 1)
            for position in range(len(ordered))
        ]
        ax.scatter(
            [index + offset for offset in offsets],
            ordered,
            color=color,
            s=42,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        median = ordered[(len(ordered) - 1) // 2] if len(ordered) % 2 else (
            ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]
        ) / 2
        ax.scatter(
            index,
            median,
            color="#222222",
            marker="D",
            s=34,
            zorder=3,
        )
        ax.text(index, 465, f"n={len(ordered)}", ha="center", va="center", fontsize=8)
    ax.set_ylabel("Length of repeated-base event (bases)")
    ax.set_ylim(0, 500)
    ax.set_yticks([0, 20, 100, 200, 300, 400, 500])
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.28, top=0.97)
    save_figure(fig, figure_dir, "repeat_behavior_vs_true_suffix")


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
        }
    )


def plot_model_comparison(root: Path, figure_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(root / "statistics_model_accuracy.csv")
        if row["model"] in NEURAL_ORDER
    ]
    values = {row["model"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y_positions = list(range(len(NEURAL_ORDER)))
    for y, model in zip(y_positions, NEURAL_ORDER):
        row = values[model]
        value = float(row["mean_genome_accuracy_percent"])
        low = value - float(row["ci_lower_percent"])
        high = float(row["ci_upper_percent"]) - value
        ax.errorbar(
            value,
            y,
            xerr=[[low], [high]],
            fmt="o",
            color=COLORS[model],
            markersize=7,
            capsize=4,
            linewidth=1.5,
        )
        ax.text(value + high + 0.25, y, f"{value:.2f}%", va="center", fontsize=9)
    ax.set_yticks(y_positions, [LABELS[m] for m in NEURAL_ORDER])
    ax.set_xlabel("Mean accuracy across genomes (%)")
    ax.set_xlim(50, 68)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figure_dir, "model_accuracy_with_ci")


def plot_distance(root: Path, figure_dir: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in read_csv(root / "accuracy_by_distance.csv"):
        if row["model"] in NEURAL_ORDER and int(row["distance_start"]) < 500:
            grouped[(row["model"], int(row["distance_start"]))].append(
                float(row["accuracy_percent"])
            )
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for model in NEURAL_ORDER:
        points = sorted(
            (distance, mean(values))
            for (candidate, distance), values in grouped.items()
            if candidate == model
        )
        ax.plot(
            [point[0] + 25 for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=COLORS[model],
            label=LABELS[model],
        )
    ax.set_xlabel("Generated position")
    ax.set_ylabel("Exact positional accuracy (%)")
    ax.set_xlim(0, 512)
    ax.set_ylim(25, 90)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, figure_dir, "accuracy_by_generation_distance")


def plot_context(root: Path, figure_dir: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in read_csv(root / "context_length_comparison.csv"):
        if row["model"] in NEURAL_ORDER:
            grouped[(row["model"], int(row["context_length"]))].append(
                float(row["accuracy_percent"])
            )
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for model in NEURAL_ORDER:
        points = sorted(
            (context, mean(values))
            for (candidate, context), values in grouped.items()
            if candidate == model
        )
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=5,
            linewidth=2,
            color=COLORS[model],
            label=LABELS[model],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1,024"])
    ax.set_xlabel("Prompt context length (bases)")
    ax.set_ylabel("Recursive accuracy (%)")
    ax.set_ylim(35, 68)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, figure_dir, "accuracy_by_context_length")


def plot_teacher_recursive(root: Path, figure_dir: Path) -> None:
    teacher_rows = {
        row["model"]: float(row["teacher_forced_exact_triplet_accuracy_percent"])
        for row in read_csv(root / "teacher_forced_metrics.csv")
        if row["model"] in NEURAL_ORDER
    }
    comparison = defaultdict(list)
    for row in read_csv(root / "model_comparison.csv"):
        if row["model"] in NEURAL_ORDER:
            comparison[row["model"]].append(
                float(row["recursive_exact_triplet_accuracy_percent"])
            )
    recursive = {model: mean(values) for model, values in comparison.items()}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y_positions = list(range(len(NEURAL_ORDER)))
    for y, model in zip(y_positions, NEURAL_ORDER):
        left = recursive[model]
        right = teacher_rows[model]
        ax.plot([left, right], [y, y], color="#999999", linewidth=2, zorder=1)
        ax.scatter(left, y, color=COLORS[model], s=45, zorder=2)
        ax.scatter(right, y, color=COLORS[model], marker="D", s=35, zorder=2)
        ax.text(left - 0.8, y, f"{left:.1f}", ha="right", va="center", fontsize=8)
        ax.text(right + 0.8, y, f"{right:.1f}", ha="left", va="center", fontsize=8)
    ax.set_yticks(y_positions, [LABELS[m] for m in NEURAL_ORDER])
    ax.set_xlabel("Exact triplet accuracy (%)")
    ax.set_xlim(30, 88)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", label="recursive", markersize=6),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#555555", label="teacher-forced", markersize=5),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.27, top=0.98)
    save_figure(fig, figure_dir, "teacher_forced_vs_recursive")


def plot_regions(root: Path, figure_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_csv(root / "results_by_region.csv"):
        if row["model"] in NEURAL_ORDER:
            grouped[(row["model"], row["region"])].append(
                float(row["accuracy_percent"]) * int(row["bases"])
            )
    denominators: dict[tuple[str, str], int] = defaultdict(int)
    for row in read_csv(root / "results_by_region.csv"):
        if row["model"] in NEURAL_ORDER:
            key = (row["model"], row["region"])
            denominators[key] += int(row["bases"])
    regions = ["IRA", "IRB", "LSC", "SSC", "boundary"]
    # Recompute weighted source aggregation for each model and seed.
    by_seed: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in read_csv(root / "results_by_region.csv"):
        if row["model"] in NEURAL_ORDER:
            by_seed[(row["model"], row["seed"], row["region"])].append(
                (float(row["accuracy_percent"]), int(row["bases"]))
            )
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, seed, region), source_rows in by_seed.items():
        total_bases = sum(bases for _, bases in source_rows)
        values[(model, region)].append(
            sum(accuracy * bases for accuracy, bases in source_rows) / total_bases
        )
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x_positions = list(range(len(regions)))
    for model in NEURAL_ORDER:
        y_values = [mean(values[(model, region)]) for region in regions]
        ax.plot(
            x_positions,
            y_values,
            marker="o",
            markersize=5,
            linewidth=2,
            color=COLORS[model],
            label=LABELS[model],
        )
    ax.set_xticks(x_positions, regions)
    ax.set_xlabel("Plastid region")
    ax.set_ylabel("Recursive accuracy (%)")
    ax.set_ylim(40, 80)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, figure_dir, "accuracy_by_plastid_region")


def write_results_draft(root: Path, figure_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Results\n\n"
        "Validation loss decreased throughout training for all four neural models, while teacher-forced per-base accuracy increased. The curves fell quickly during the first several epochs and then flattened, with smaller changes near the end of training. The 26.98M S4D model remained lowest on validation loss, followed by the 16.60M S4D model, the Transformer, and the 4.13M S4D model. Their best validation losses were 0.9761, 1.0629, 1.1621, and 1.4113, respectively. The accuracy panel shows the same ordering, but these curves came from records used in the training-data split and were not the final generalization test. The curves were still improving slowly near the end of the 50-epoch budget, so the comparison used the same maximum training budget rather than selecting different stopping points for different model sizes.\n\n"
        f"![Validation loss and teacher-forced accuracy across training]({figure_dir.name}/training_validation_progress.png)\n\n"
        "The final evaluation used 20 plastid genomes that were absent from the training records by accession and sequence-hash checks. Ten also belonged to genera not represented in the training records. Every model received the same circular, overlapping windows, 512-base prompts, and 512-base targets. Neural models used sampled decoding at temperature 0.8 with seeds 13, 17, and 23.\n\n"
        "The confidence-interval plot shows a consistent increase in recursive accuracy with model size. The 26.98M S4D model had the highest genome-level mean at 63.38%, followed by the 16.60M S4D model at 62.44%, the Transformer at 59.36%, and the 4.13M S4D model at 57.03%. The intervals for the S4D models were separated from the smaller-model results, while the difference between the two larger S4D models was narrower. The 26.98M model exceeded the 16.60M model by 0.93 percentage points, with a 95% genome-level bootstrap interval of 0.61 to 1.32 points. The 16.60M S4D model exceeded the matched Transformer by 3.08 points, with an interval of 2.24 to 4.02 points. The largest model was best on 19 of the 20 genomes. The gain from 4.13M to 16.60M was larger than the gain from 16.60M to 26.98M, so increasing width helped, but the improvement began to level off at the largest size.\n\n"
        f"![Model accuracy with genome-level bootstrap intervals]({figure_dir.name}/model_accuracy_with_ci.png)\n\n"
        "On the same untouched panel, teacher-forced scoring gave base-normalized perplexities of 1.448 for the 4.13M S4D model, 1.280 for the 16.60M S4D model, 1.252 for the 26.98M S4D model, and 1.316 for the Transformer. The dot plot shows the same ordering as recursive accuracy, with the 26.98M S4D model assigning the highest probability to the observed test sequences. Lower perplexity means that the model assigned greater probability to the observed test sequence when the correct history was supplied. This ranking agrees with the recursive accuracy ranking, but the two measurements answer different questions: perplexity measures sequence likelihood under the correct history, while recursive accuracy measures what happens after the model begins consuming its own outputs.\n\n"
        f"![Untouched-test perplexity]({figure_dir.name}/untouched_test_perplexity.png)\n\n"
        "The distance plot shows a downward curve for every model. Accuracy was measured every 50 generated bases through position 475; the final 12-base partial section was omitted. Accuracy decreased as the model generated farther from the prompt. For the 26.98M S4D model, accuracy declined from 83.77% at the beginning to 50.65% at the end. The corresponding declines were from 82.54% to 50.00% for the 16.60M S4D model, from 79.76% to 47.35% for the Transformer, and from 75.59% to 46.28% for the 4.13M model. The lines remain ordered for most of the continuation, so model size improved the starting point and preserved some advantage at longer distances, but none removed the decline over a 512-base continuation. This pattern is consistent with error accumulation: an early incorrect triplet changes the history used for later predictions.\n\n"
        f"![Accuracy by generated distance]({figure_dir.name}/accuracy_by_generation_distance.png)\n\n"
        "Global alignment identity was higher than exact positional accuracy for every neural model. This means some generated continuations were still sequence-similar to the true suffix even when bases did not match at the exact same positions. The 26.98M S4D model had the highest alignment identity at 84.96%, followed by the 16.60M S4D model at 84.03%, the Transformer at 80.24%, and the 4.13M S4D model at 78.10%. Alignment identity is treated as a secondary measure because the generation task still asks for the correct next 512 bases in order, but it helps separate completely wrong sequence from shifted or partially aligned sequence.\n\n"
        f"![Exact accuracy and global alignment identity]({figure_dir.name}/exact_accuracy_vs_alignment_identity.png)\n\n"
        "The dumbbell plot makes the teacher-forced versus recursive gap visible for each model. The 26.98M model reached 82.93% exact triplet accuracy with the true preceding sequence, compared with 46.67% during recursive generation. The corresponding recursive and teacher-forced values were 45.18% and 81.32% for the 16.60M model, 41.23% and 79.21% for the Transformer, and 38.03% and 72.17% for the 4.13M model. Every model had a large gap between the two markers, including the largest model. Thus, the models learned useful local sequence statistics, but maintaining a correct history during long free generation remained harder than predicting from an uncorrupted history.\n\n"
        f"![Teacher-forced and recursive triplet accuracy]({figure_dir.name}/teacher_forced_vs_recursive.png)\n\n"
        "The context-length plot rises sharply from 128 to 256 bases for every neural model and then bends toward a plateau. Increasing the prompt from 128 to 256 bases produced the largest improvement, while changing the prompt from 512 to 1,024 bases had almost no effect. For the 26.98M S4D model, recursive accuracy was 53.16%, 61.71%, 63.67%, and 63.63% at context lengths of 128, 256, 512, and 1,024 bases. The 16.60M model changed from 62.94% to 63.02% between 512 and 1,024 bases, and the Transformer changed from 59.52% to 59.96%. These results show that the current model benefited from several hundred bases of context, but they do not demonstrate a further gain from the full 1,024-base input.\n\n"
        f"![Accuracy by prompt context length]({figure_dir.name}/accuracy_by_context_length.png)\n\n"
        "The region plot shows higher accuracy in the inverted-repeat categories than in the single-copy categories for all four neural models. For the 26.98M S4D model, accuracy was approximately 74% in IRA and IRB, approximately 58% in LSC and SSC, and approximately 54% in boundary windows. The model lines are relatively close within each region, but the separation between repeat and single-copy regions is larger than the difference between models in some categories. This may reflect the repeated and more conserved sequence structure of the inverted repeats. Region labels came from GenBank annotations when available and from sequence-based inference otherwise, so this comparison is treated as secondary rather than as a definitive feature-level conclusion.\n\n"
        f"![Accuracy by plastid region]({figure_dir.name}/accuracy_by_plastid_region.png)\n\n"
        "The repeat plot is conditioned on events of at least 20 bases, so it focuses on the rare long-repeat cases rather than the ordinary 3-8 base runs. Because some groups contain only a few events, the plot shows every event as a dot instead of using a box plot. The diamond in each group marks the median. S4D 4.13M had no events in this range and is therefore omitted from the figure. The event counts were 4 for S4D 16.60M, 5 for S4D 26.98M, 8 for the Transformer, and 59 in the true suffixes. The Transformer was closest to the real suffixes for the central event length: both had a median of 25 bases, and their middle 50% ranges were similar at 20.75-27.5 bases for the Transformer and 22-31.5 bases for the true suffixes. However, the Transformer did not reproduce the real long tail. The true suffixes included events of 123, 258, and 379 bases, while the Transformer maximum was 50 bases. This means the Transformer was more similar for typical long-repeat events but still failed to model the rare extreme repeats. Each neural model produced 35,877 continuations from the 11,959 windows across three seeds, while the real comparison used the 11,959 unique true suffixes. For runs longer than 100 bases, S4D 4.13M had 0 events (0%), S4D 16.60M had 2 events (0.0056%), S4D 26.98M had 4 events (0.0111%), and the Transformer had 0 events (0%). The real suffixes had 3 windows with runs above 100 bases (0.0251%). The 16.60M events did not match real long-repeat windows. Two of the four 26.98M events matched real long-repeat windows, while the other two occurred in an adjacent window where the true suffix had only an 8-base run. One 26.98M output exceeded 300 bases, and none reached 400 bases; the maximum was 354 bases. The largest generated run occurred in the same A-rich test region that contained the longest real run, so this event extended a real low-complexity pattern rather than creating the entire run from unrelated sequence. The count-based baselines produced approximately 31% accuracy and often generated a single repeated base, so they provide a lower bound rather than a competitive sequence model. These results show why exact accuracy alone is not enough to describe generation quality.\n\n"
        f"![Repeat behavior compared with true test suffixes]({figure_dir.name}/repeat_behavior_vs_true_suffix.png)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-test figures and results text.")
    parser.add_argument("--input-dir", default="outputs/final_test")
    parser.add_argument("--figure-dir", default="reports/final_test/figures")
    parser.add_argument("--results-file", default="reports/final_test_results.md")
    args = parser.parse_args()
    root = Path(args.input_dir)
    figure_dir = Path(args.figure_dir)
    configure_plot()
    plot_training_progress(figure_dir)
    plot_test_perplexity(root, figure_dir)
    plot_test_perplexity_bar(root, figure_dir)
    plot_alignment_identity(root, figure_dir)
    plot_repeat_behavior(root, figure_dir)
    plot_model_comparison(root, figure_dir)
    plot_distance(root, figure_dir)
    plot_context(root, figure_dir)
    plot_teacher_recursive(root, figure_dir)
    plot_regions(root, figure_dir)
    write_results_draft(root, figure_dir, Path(args.results_file))
    print(f"Figures written to {figure_dir}")
    print(f"Results draft written to {args.results_file}")


if __name__ == "__main__":
    main()
