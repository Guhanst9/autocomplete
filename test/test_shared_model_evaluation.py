import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.baselines.checkpoint import save_baseline_checkpoint
from src.dna.checkpoint import build_model_from_config, load_model
from src.dna.data import DnaTokenizer
from src.dna.prediction import TripletCodec
import src.evaluation.shared as shared_evaluation
from src.evaluation.shared import (
    freeze_evaluation,
    run_shared_evaluation,
    verify_evaluation_lock,
)
from src.evaluation.test_panel import file_sha256, sequence_sha256


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def prepare_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    sequence = "ACGT" * 8
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "TEST_1.1.fasta").write_text(f">TEST_1.1 Rosa test\n{sequence}\n")

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "accession",
                "species",
                "genus",
                "length",
                "source",
                "source_type",
                "sequence_sha256",
                "reverse_complement_sha256",
                "labels",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "accession": "TEST_1.1",
                "species": "Rosa test",
                "genus": "Rosa",
                "length": len(sequence),
                "source": "test",
                "source_type": "ncbi-accession",
                "sequence_sha256": sequence_sha256(sequence),
                "reverse_complement_sha256": sequence_sha256(sequence),
                "labels": "unseen-accession",
            }
        )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"training_fasta_sha256": "training-hash"}))
    panel_config = tmp_path / "panel.yaml"
    write_yaml(
        panel_config,
        {
            "panel_role": "untouched-test",
            "manifest": str(manifest),
            "frozen_manifest_sha256": file_sha256(manifest),
            "metadata": str(metadata),
            "download_dir": str(downloads),
        },
    )

    tables = {str(order): {} for order in range(1, 7)}
    tables["1"]["T"] = [0] * 64
    tables["1"]["T"][TripletCodec().encode("ACG")] = 10
    baseline = tmp_path / "baseline.json.gz"
    save_baseline_checkpoint(
        {
            "format_version": 1,
            "prediction_unit": "triplet",
            "triplet_vocabulary": TripletCodec().triplets,
            "base_counts": [10, 1, 1, 1],
            "triplet_counts": [10] + [0] * 63,
            "markov_counts": tables,
            "training": {
                "fasta_sha256": "training-hash",
                "train_record_fingerprint": "train-fingerprint",
                "validation_record_fingerprint": "validation-fingerprint",
            },
        },
        baseline,
    )
    tokenizer = DnaTokenizer()
    neural_config = {
        "model_type": "transformer",
        "prediction_unit": "triplet",
        "output_vocab_size": 64,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "max_length": 32,
    }
    neural = tmp_path / "neural.pt"
    torch.save(
        {
            "model_state_dict": build_model_from_config(neural_config, tokenizer).state_dict(),
            "model_type": "transformer",
            "model_config": neural_config,
            "prediction_unit": "triplet",
            "output_vocab": TripletCodec().triplets,
            "tokenizer_vocab": tokenizer.vocab,
            "epoch": 0,
            "data_fingerprints": {
                "train_records": "train-fingerprint",
                "val_records": "validation-fingerprint",
                "val_windows": "validation-windows",
            },
        },
        neural,
    )
    models_config = tmp_path / "models.yaml"
    write_yaml(
        models_config,
        {
            "frozen": True,
            "evaluation": {
                "prompt_length": 8,
                "generation_length": 8,
                "stride": 4,
                "circular": True,
                "decoding_mode": "sampled",
                "temperature": 0.8,
                "seed": 13,
                "sampling_seeds": [13, 17],
                "baseline_smoothing": 1.0,
                "edge_bases": 4,
                "distance_bin_width": 2,
                "alignment_scores": {
                    "match": 2.0,
                    "mismatch": -1.0,
                    "gap_open": -2.0,
                    "gap_extension": -0.5,
                },
                "statistical_baseline": "markov-1",
                "bootstrap_replicates": 100,
                "bootstrap_confidence": 0.95,
                "bootstrap_seed": 13,
                "batch_size": 2,
                "context_lengths": [4, 8],
                "context_targets_per_genome": 2,
            },
            "models": [
                {
                    "name": "most-common-base",
                    "kind": "baseline",
                    "checkpoint": str(baseline),
                    "method": "most-common-base",
                },
                {
                    "name": "markov-1",
                    "kind": "baseline",
                    "checkpoint": str(baseline),
                    "method": "markov",
                    "order": 1,
                },
                {
                    "name": "tiny-transformer",
                    "kind": "neural",
                    "checkpoint": str(neural),
                },
            ],
        },
    )
    return panel_config, models_config, manifest


def test_freeze_and_shared_evaluation_reuses_loaded_neural_model(tmp_path, monkeypatch):
    panel_config, models_config, _ = prepare_files(tmp_path)
    output = tmp_path / "output"
    lock_path = freeze_evaluation(str(panel_config), str(models_config), str(output))
    lock = json.loads(lock_path.read_text())
    assert lock["windows"] == 7

    load_calls = []

    def counting_load(checkpoint):
        load_calls.append(checkpoint)
        return load_model(checkpoint)

    monkeypatch.setattr(shared_evaluation, "load_model", counting_load)
    summary_path = run_shared_evaluation(str(panel_config), str(models_config), str(output))
    assert len(load_calls) == 1
    with summary_path.open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    assert [row["model"] for row in summaries] == [
        "most-common-base",
        "markov-1",
        "tiny-transformer",
        "tiny-transformer",
    ]

    result_paths = [
        output / "models" / name / "seed_deterministic" / "TEST_1.1_windows.csv"
        for name in ("most-common-base", "markov-1")
    ]
    result_rows = []
    for path in result_paths:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all(len(row["generated_suffix"]) == 8 for row in rows)
        result_rows.append(rows)
    assert [row["window_start"] for row in result_rows[0]] == [
        row["window_start"] for row in result_rows[1]
    ]
    assert [row["true_suffix"] for row in result_rows[0]] == [
        row["true_suffix"] for row in result_rows[1]
    ]

    for name in (
        "results_by_genome.csv",
        "results_by_plant_group.csv",
        "results_by_region.csv",
        "context_length_comparison.csv",
        "context_length_by_genome.csv",
        "teacher_forced_metrics.csv",
        "accuracy_by_distance.csv",
        "accuracy_by_distance_and_genome.csv",
        "statistics_genome_seed_averages.csv",
        "statistics_model_accuracy.csv",
        "statistics_paired_differences.csv",
        "statistics_vs_baseline.csv",
    ):
        assert (output / name).exists()

    with (output / "model_comparison.csv").open(newline="") as handle:
        comparison = list(csv.DictReader(handle))
    assert all("recursive_exact_triplet_accuracy_percent" in row for row in comparison)
    assert all("mean_global_alignment_identity_percent" in row for row in comparison)

    with (output / "teacher_forced_metrics.csv").open(newline="") as handle:
        teacher_rows = list(csv.DictReader(handle))
    assert len(teacher_rows) == 3
    assert all(float(row["base_normalized_perplexity"]) > 0 for row in teacher_rows)

    with (output / "results_by_plant_group.csv").open(newline="") as handle:
        group_rows = list(csv.DictReader(handle))
    assert {row["plant_group"] for row in group_rows} == {"Rosa"}

    context_rows = {}
    for context_length in (4, 8):
        path = (
            output
            / "context"
            / "markov-1"
            / "seed_deterministic"
            / f"context_{context_length}"
            / "TEST_1.1_windows.csv"
        )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all(len(row["prompt"]) == context_length for row in rows)
        context_rows[context_length] = {
            (row["target_start"], row["true_suffix"]) for row in rows
        }
    assert len(set(map(frozenset, context_rows.values()))) == 1

    first_mtime = result_paths[0].stat().st_mtime_ns
    (output / "run_complete.json").unlink()
    run_shared_evaluation(str(panel_config), str(models_config), str(output))
    assert result_paths[0].stat().st_mtime_ns == first_mtime

    with pytest.raises(FileExistsError, match="must not be rerun"):
        run_shared_evaluation(str(panel_config), str(models_config), str(output))


def test_frozen_evaluation_rejects_changed_config(tmp_path):
    panel_config, models_config, _ = prepare_files(tmp_path)
    output = tmp_path / "output"
    freeze_evaluation(str(panel_config), str(models_config), str(output))
    models_config.write_text(models_config.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen file changed"):
        verify_evaluation_lock(str(panel_config), str(models_config), str(output))


def test_freeze_requires_explicit_finalization(tmp_path):
    panel_config, models_config, _ = prepare_files(tmp_path)
    config = yaml.safe_load(models_config.read_text())
    config["frozen"] = False
    write_yaml(models_config, config)
    with pytest.raises(ValueError, match="frozen: true"):
        freeze_evaluation(str(panel_config), str(models_config), str(tmp_path / "output"))
