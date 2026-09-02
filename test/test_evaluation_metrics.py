import math

import pytest
import torch

from src.dna.checkpoint import build_model_from_config
from src.dna.data import DnaTokenizer
from src.dna.prediction import TripletCodec
from src.evaluation.metrics import (
    alignment_identity,
    baseline_teacher_metrics,
    distance_bin_rows,
    make_global_aligner,
    neural_teacher_metrics,
    recursive_triplet_accuracy,
)
from src.sliding_eval.windows import SlidingWindow


def result_row(generated: str, truth: str) -> dict[str, str]:
    return {
        "model": "test",
        "seed": "13",
        "generated_suffix": generated,
        "true_suffix": truth,
    }


def test_recursive_triplets_follow_emitted_three_base_blocks():
    correct, total = recursive_triplet_accuracy([result_row("AAACCC", "AAAGGG")])
    assert (correct, total) == (1, 2)


def test_distance_bins_use_exact_positions():
    rows = [result_row("AAAACC", "AAAAGG")]
    bins = distance_bin_rows(rows, ["model", "seed"], width=2)
    assert [row["accuracy_percent"] for row in bins] == [100.0, 100.0, 0.0]


def test_global_alignment_can_recover_a_shift():
    aligner = make_global_aligner()
    aligned = alignment_identity("TACGT", "ACGTT", aligner)
    exact = 100 * sum(a == b for a, b in zip("TACGT", "ACGTT")) / 5
    assert aligned > exact


def test_count_teacher_metrics_are_finite_with_smoothing():
    window = SlidingWindow(
        window_start=0,
        prompt_start=0,
        prompt_end=3,
        target_start=4,
        target_end=9,
        region="unknown",
        prompt="ACGT",
        true_suffix="TTTAAA",
    )
    checkpoint = {
        "triplet_vocabulary": [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"],
        "base_counts": [10, 5, 5, 5],
        "triplet_counts": [0] * 64,
        "markov_counts": {str(order): {} for order in range(1, 7)},
    }
    metrics = baseline_teacher_metrics(
        {"method": "triplet-frequency"},
        checkpoint,
        [window],
        smoothing=1.0,
    )
    assert metrics["teacher_forced_triplets"] == 4
    assert math.isfinite(metrics["base_normalized_perplexity"])


@pytest.mark.parametrize("model_type", ["s4d", "transformer"])
def test_neural_teacher_metrics_score_triplet_checkpoints(tmp_path, model_type):
    tokenizer = DnaTokenizer()
    config = {
        "model_type": model_type,
        "prediction_unit": "triplet",
        "output_vocab_size": 64,
        "d_model": 16,
        "n_layers": 1,
        "dropout": 0.0,
        "l_max": 16,
        "max_length": 16,
    }
    if model_type == "s4d":
        config.update({"d_state": 4, "kernel_type": "diag", "model_variant": "s4d_v2"})
    else:
        config.update({"n_heads": 4, "ffn_dim": 32})
    model = build_model_from_config(config, tokenizer)
    checkpoint = tmp_path / f"{model_type}.pt"
    torch.save(
        {
            "model_type": model_type,
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "prediction_unit": "triplet",
            "output_vocab": TripletCodec().triplets,
            "tokenizer_vocab": tokenizer.vocab,
        },
        checkpoint,
    )
    window = SlidingWindow(
        window_start=0,
        prompt_start=0,
        prompt_end=7,
        target_start=8,
        target_end=15,
        region="unknown",
        prompt="ACGTACGT",
        true_suffix="TGCATGCA",
    )
    metrics = neural_teacher_metrics(str(checkpoint), [window], batch_size=1)
    assert metrics["teacher_forced_triplets"] == 6
    assert math.isfinite(metrics["teacher_forced_triplet_cross_entropy"])
    assert math.isfinite(metrics["base_normalized_perplexity"])
