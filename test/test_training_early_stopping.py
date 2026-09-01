import pytest

from src.dna.training import early_stopping_step, run_training


def test_early_stopping_counts_adjacent_epoch_plateaus():
    previous, plateau_epochs = early_stopping_step(1.0, 0, 0.996, 0.005)
    assert previous == 0.996
    assert plateau_epochs == 1

    previous, plateau_epochs = early_stopping_step(previous, plateau_epochs, 0.988, 0.005)
    assert previous == 0.988
    assert plateau_epochs == 0

    previous, plateau_epochs = early_stopping_step(previous, plateau_epochs, 0.991, 0.005)
    assert previous == 0.991
    assert plateau_epochs == 1


def test_early_stopping_counts_regressions_and_skips_initial_epoch():
    previous, plateau_epochs = early_stopping_step(None, 0, 1.0, 0.002)
    assert previous == 1.0
    assert plateau_epochs == 0

    previous, plateau_epochs = early_stopping_step(previous, plateau_epochs, 1.001, 0.002)
    assert previous == 1.001
    assert plateau_epochs == 1

    previous, plateau_epochs = early_stopping_step(previous, plateau_epochs, 1.004, 0.002)
    assert previous == 1.004
    assert plateau_epochs == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_additional_epochs": 0}, "max_additional_epochs"),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"early_stopping_min_delta": -0.001}, "early_stopping_min_delta"),
        (
            {"early_stopping_previous_val_loss": float("inf")},
            "early_stopping_previous_val_loss",
        ),
    ],
)
def test_training_control_validation_happens_before_data_loading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_training(
            preset_name="smoke",
            fasta_file="not-used.fna.gz",
            output_dir="not-used",
            resume=None,
            holdout_accession="NC_TEST",
            **kwargs,
        )
