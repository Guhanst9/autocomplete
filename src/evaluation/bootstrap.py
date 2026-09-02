import hashlib
import random
from itertools import combinations
from statistics import fmean


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("percentile probability must be between zero and one")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: list[float],
    replicates: int,
    confidence: float,
    seed: int,
    label: str,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one genome")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between zero and one")
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    sampled_means = [
        fmean(rng.choice(values) for _ in values)
        for _ in range(replicates)
    ]
    tail = (1 - confidence) / 2
    return (
        fmean(values),
        percentile(sampled_means, tail),
        percentile(sampled_means, 1 - tail),
    )


def average_seeds_by_genome(genome_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in genome_rows:
        key = (row["model"], row["kind"], row["accession"])
        grouped.setdefault(key, []).append(float(row["accuracy_percent"]))
    return [
        {
            "model": model,
            "kind": kind,
            "accession": accession,
            "sampling_runs": len(values),
            "accuracy_percent": fmean(values),
        }
        for (model, kind, accession), values in sorted(grouped.items())
    ]


def bootstrap_model_comparisons(
    genome_rows: list[dict],
    baseline_name: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    averaged = average_seeds_by_genome(genome_rows)
    by_model: dict[str, dict[str, float]] = {}
    kinds = {}
    for row in averaged:
        model = row["model"]
        kinds.setdefault(model, row["kind"])
        if kinds[model] != row["kind"]:
            raise ValueError(f"model kind changed across rows: {model}")
        by_model.setdefault(model, {})[row["accession"]] = float(row["accuracy_percent"])
    if baseline_name not in by_model:
        raise ValueError(f"statistical baseline is missing: {baseline_name}")

    accession_sets = {model: set(values) for model, values in by_model.items()}
    expected = accession_sets[baseline_name]
    for model, accessions in accession_sets.items():
        if accessions != expected:
            raise ValueError(f"genome sets do not match for paired comparison: {model}")
    accessions = sorted(expected)

    model_rows = []
    for model in sorted(by_model):
        values = [by_model[model][accession] for accession in accessions]
        mean, lower, upper = bootstrap_mean_interval(
            values, replicates, confidence, seed, f"model:{model}"
        )
        model_rows.append(
            {
                "model": model,
                "kind": kinds[model],
                "genomes": len(values),
                "mean_genome_accuracy_percent": mean,
                "ci_lower_percent": lower,
                "ci_upper_percent": upper,
            }
        )

    paired_rows = []
    for model_a, model_b in combinations(sorted(by_model), 2):
        differences = [
            by_model[model_a][accession] - by_model[model_b][accession]
            for accession in accessions
        ]
        mean, lower, upper = bootstrap_mean_interval(
            differences,
            replicates,
            confidence,
            seed,
            f"pair:{model_a}:{model_b}",
        )
        paired_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "genomes": len(differences),
                "mean_paired_difference_points": mean,
                "ci_lower_points": lower,
                "ci_upper_points": upper,
            }
        )

    baseline_rows = []
    for model in sorted(by_model):
        if model == baseline_name:
            continue
        differences = [
            by_model[model][accession] - by_model[baseline_name][accession]
            for accession in accessions
        ]
        mean, lower, upper = bootstrap_mean_interval(
            differences,
            replicates,
            confidence,
            seed,
            f"baseline:{model}:{baseline_name}",
        )
        baseline_rows.append(
            {
                "model": model,
                "baseline": baseline_name,
                "genomes": len(differences),
                "mean_improvement_points": mean,
                "ci_lower_points": lower,
                "ci_upper_points": upper,
            }
        )
    return averaged, model_rows, paired_rows, baseline_rows
