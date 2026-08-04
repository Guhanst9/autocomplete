import _path  # noqa: F401

from src.dna.training import (
    gc_fraction,
    kmer_diversity,
    quality_score,
    sequence_entropy,
)


def test_sequence_quality_metrics():
    assert gc_fraction("AACC") == 0.5
    assert gc_fraction("") == 0.0
    assert sequence_entropy("AAAA") == 0.0
    assert round(sequence_entropy("ACGT"), 6) == 2.0
    assert kmer_diversity("AAAAAAAA", k=8) == 1.0
    assert kmer_diversity("AAAAAAAAA", k=8) == 0.5
    assert kmer_diversity("ACGT", k=8) == 0.0


def test_quality_score_penalizes_bad_generation():
    metrics = {
        "accuracy": 0.70,
        "low_complexity_fraction": 0.10,
        "runs_over_20_fraction": 0.05,
        "mean_gc_difference": 0.20,
        "n_count": 0,
    }
    assert abs(quality_score(metrics) - 0.35) < 1e-9

    metrics["n_count"] = 1
    assert quality_score(metrics) == float("-inf")


if __name__ == "__main__":
    test_sequence_quality_metrics()
    test_quality_score_penalizes_bad_generation()
    print("Quality metric tests passed.")
