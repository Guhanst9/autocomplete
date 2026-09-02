import pytest

from src.evaluation.bootstrap import (
    average_seeds_by_genome,
    bootstrap_mean_interval,
    bootstrap_model_comparisons,
)


def genome_row(model: str, accession: str, accuracy: float, seed: str = "13") -> dict:
    return {
        "model": model,
        "kind": "baseline" if model == "baseline" else "neural",
        "accession": accession,
        "seed": seed,
        "accuracy_percent": accuracy,
    }


def test_seed_results_are_averaged_within_genome():
    rows = [
        genome_row("model", "A", 60.0, "13"),
        genome_row("model", "A", 80.0, "17"),
        genome_row("model", "B", 50.0, "13"),
        genome_row("model", "B", 70.0, "17"),
    ]
    averaged = average_seeds_by_genome(rows)
    assert [row["accuracy_percent"] for row in averaged] == [70.0, 60.0]
    assert all(row["sampling_runs"] == 2 for row in averaged)


def test_bootstrap_is_reproducible_and_pairs_whole_genomes():
    rows = [
        genome_row("baseline", "A", 30.0, "deterministic"),
        genome_row("baseline", "B", 40.0, "deterministic"),
        genome_row("model", "A", 50.0),
        genome_row("model", "B", 70.0),
    ]
    first = bootstrap_model_comparisons(rows, "baseline", 500, 0.95, 13)
    second = bootstrap_model_comparisons(rows, "baseline", 500, 0.95, 13)
    assert first == second
    improvement = first[3][0]
    assert improvement["mean_improvement_points"] == 25.0
    assert improvement["ci_lower_points"] == 20.0
    assert improvement["ci_upper_points"] == 30.0


def test_bootstrap_rejects_unpaired_genome_sets():
    rows = [
        genome_row("baseline", "A", 30.0),
        genome_row("model", "B", 50.0),
    ]
    with pytest.raises(ValueError, match="genome sets do not match"):
        bootstrap_model_comparisons(rows, "baseline", 100, 0.95, 13)


def test_bootstrap_mean_validates_configuration():
    with pytest.raises(ValueError, match="replicates"):
        bootstrap_mean_interval([1.0], 0, 0.95, 13, "test")
