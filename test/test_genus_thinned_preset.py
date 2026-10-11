from dataclasses import asdict

import _path  # noqa: F401

from src.dna.training import PRESETS, parameter_count, build_sequence_model
from src.dna.data import DnaTokenizer


def test_genus_thinned_large_preset_matches_experiment():
    preset = PRESETS["genus-thinned-large-50"]
    config = asdict(preset)

    assert config["d_model"] == 512
    assert config["d_state"] == 64
    assert config["n_layers"] == 10
    assert config["l_max"] == 1024
    assert config["windows_per_record"] == 16
    assert config["max_windows"] == 60_000
    assert config["epochs"] == 50
    assert config["batch_size"] == 8
    assert config["lr"] == 3e-4
    assert config["dropout"] == 0.1
    assert config["seed"] == 13
    assert config["recovery_corruption_mode"] == "independent"
    assert config["resample_train_windows"] is True

    model = build_sequence_model(
        tokenizer=DnaTokenizer(),
        preset=preset,
        prediction_unit="triplet",
    )
    assert parameter_count(model) == 26_978_816
