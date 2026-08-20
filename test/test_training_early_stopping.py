import pytest

from src.dna.training import early_stopping_step, run_training


def test_early_stopping_counts_against_last_significant_improvement():
    reference, failures = early_stopping_step(1.0, 0, 0.996, 0.005)
    assert reference == 1.0
    assert failures == 1

    reference, failures = early_stopping_step(reference, failures, 0.994, 0.005)
    assert reference == 0.994
    assert failures == 0

    reference, failures = early_stopping_step(reference, failures, 0.991, 0.005)
    assert reference == 0.994
    assert failures == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_additional_epochs": 0}, "max_additional_epochs"),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"early_stopping_min_delta": -0.001}, "early_stopping_min_delta"),
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
