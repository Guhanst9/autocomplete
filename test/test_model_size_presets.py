import _path  # noqa: F401

from dataclasses import asdict

from benchmark_model_sizes import select_common_batch
from src.dna.data import DnaTokenizer
from src.dna.training import PRESETS, build_sequence_model, parameter_count


SIZE_PRESETS = ("size-small", "size-current", "size-large")


def test_size_presets_share_all_training_settings_except_width():
    configs = []
    for name in SIZE_PRESETS:
        config = asdict(PRESETS[name])
        config.pop("name")
        config.pop("d_model")
        configs.append(config)

    assert configs[0] == configs[1] == configs[2]
    shared = configs[0]
    assert shared["d_state"] == 64
    assert shared["n_layers"] == 10
    assert shared["l_max"] == 1024
    assert shared["windows_per_record"] == 4
    assert shared["max_windows"] == 60000
    assert shared["epochs"] == 20
    assert shared["batch_size"] == 2
    assert shared["lr"] == 3e-4
    assert shared["dropout"] == 0.1
    assert shared["seed"] == 13
    assert shared["stride"] == 256
    assert shared["resample_train_windows"] is True
    assert shared["recovery_enabled"] is True
    assert shared["recovery_corruption_mode"] == "independent"
    assert shared["recovery_start_probability"] == 0.02
    assert shared["recovery_max_probability"] == 0.10
    assert shared["homopolymer_loss_weight"] == 0.02


def test_size_presets_have_expected_widths_and_parameter_counts():
    tokenizer = DnaTokenizer(include_n=False)
    expected = {
        "size-small": (196, 4_134_228),
        "size-current": (400, 16_597_200),
        "size-large": (512, 26_978_816),
    }
    for name, (width, count) in expected.items():
        preset = PRESETS[name]
        model = build_sequence_model(tokenizer, preset, prediction_unit="triplet")
        assert preset.d_model == width
        assert parameter_count(model) == count


def test_common_batch_selection_requires_all_models_and_headroom():
    rows = []
    for batch_size, speed, memory in ((2, 10.0, 5000.0), (4, 18.0, 9000.0)):
        for name in SIZE_PRESETS:
            rows.append(
                {
                    "preset": name,
                    "batch_size": batch_size,
                    "status": "success",
                    "examples_per_second": speed,
                    "peak_reserved_memory_mib": memory,
                    "peak_nvidia_smi_memory_mib": memory,
                }
            )
    for name in SIZE_PRESETS[:-1]:
        rows.append(
            {
                "preset": name,
                "batch_size": 8,
                "status": "success",
                "examples_per_second": 24.0,
                "peak_reserved_memory_mib": 15000.0,
                "peak_nvidia_smi_memory_mib": 15000.0,
            }
        )
    rows.append({"preset": "size-large", "batch_size": 8, "status": "oom"})

    recommendation = select_common_batch(rows, total_memory_mib=23028.0)
    assert recommendation["selected"]["batch_size"] == 4
