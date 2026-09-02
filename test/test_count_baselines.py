import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from fit_dna_baselines import fit_baseline_checkpoint
from src.baselines.checkpoint import load_baseline_checkpoint, save_baseline_checkpoint
from src.baselines.frequency import (
    fit_frequency_counts,
    generate_frequency,
    most_common_base,
    most_common_triplet,
)
from src.baselines.markov import fit_count_baselines, generate_markov, predict_triplet


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def test_frequency_baselines_and_exact_generation_length():
    base_counts, triplet_counts = fit_frequency_counts(["AAACGTAAA"])
    assert most_common_base(base_counts) == "A"
    assert most_common_triplet(triplet_counts) == "AAA"
    assert generate_frequency("most-common-base", 7, base_counts, triplet_counts) == "A" * 7
    assert generate_frequency("triplet-frequency", 7, base_counts, triplet_counts) == "AAAAAAA"


def test_markov_prediction_and_shorter_context_backoff():
    _, triplet_counts, tables = fit_count_baselines(["ACGTACGTACGT"])
    assert predict_triplet("ACGTAC", 6, tables, triplet_counts) == "GTA"
    assert predict_triplet("TTAC", 6, tables, triplet_counts) == "GTA"
    generated = generate_markov("ACGTAC", 7, 6, tables, triplet_counts)
    assert generated == "GTACGTA"


def test_optimized_markov_counts_match_naive_reference():
    rng = random.Random(13)
    sequences = [
        "".join(rng.choice("ACGT") for _ in range(length))
        for length in (3, 4, 8, 9, 17, 51)
    ]
    _, triplet_counts, tables = fit_count_baselines(sequences)
    naive = {str(order): defaultdict(lambda: [0] * 64) for order in range(1, 7)}
    triplets = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    triplet_ids = {triplet: index for index, triplet in enumerate(triplets)}
    expected_triplets = [0] * 64
    for sequence in sequences:
        for target_start in range(len(sequence) - 2):
            target_id = triplet_ids[sequence[target_start : target_start + 3]]
            expected_triplets[target_id] += 1
            for order in range(1, min(6, target_start) + 1):
                context = sequence[target_start - order : target_start]
                naive[str(order)][context][target_id] += 1
    assert triplet_counts == expected_triplets
    assert tables == {order: dict(rows) for order, rows in naive.items()}


def test_baseline_checkpoint_gzip_round_trip(tmp_path):
    base_counts, triplet_counts, tables = fit_count_baselines(["ACGTACGTACGT"])
    checkpoint = {
        "format_version": 1,
        "prediction_unit": "triplet",
        "triplet_vocabulary": [
            a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
        ],
        "base_counts": base_counts,
        "triplet_counts": triplet_counts,
        "markov_counts": tables,
        "training": {},
    }
    path = tmp_path / "counts.json.gz"
    save_baseline_checkpoint(checkpoint, path)
    with gzip.open(path, "rt") as handle:
        assert json.load(handle)["format_version"] == 1
    assert load_baseline_checkpoint(path) == checkpoint


def test_full_fit_excludes_holdout_and_validation_records(tmp_path):
    fasta = tmp_path / "training.fasta"
    write_fasta(
        fasta,
        [
            ("KEEP_1.1 Rosa one", "AAAAAA"),
            ("KEEP_2.1 Rosa two", "CCCCCC"),
            ("NC_053550.1 Rosa holdout", "TTTTTT"),
        ],
    )
    checkpoint = fit_baseline_checkpoint(str(fasta), "NC_053550.1", 0.5, 13)
    assert checkpoint["training"]["train_records"] == 1
    assert checkpoint["training"]["validation_records"] == 1
    assert sum(checkpoint["base_counts"]) == 6
    assert checkpoint["base_counts"][3] == 0
