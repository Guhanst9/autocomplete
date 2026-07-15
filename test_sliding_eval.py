import csv
import tempfile
from argparse import Namespace

from src.sliding_eval.cli import resolve_decoding
from src.sliding_eval.fasta import PlastidRecord
from src.sliding_eval.generation import gc_difference_percent, longest_homopolymer_run
from src.sliding_eval.windows import SlidingWindow, write_windows_csv


def decoding_args(mode, sample=False, penalty=None, ngram=None):
    return Namespace(
        decoding_mode=mode,
        sample=sample,
        repetition_penalty=penalty,
        no_repeat_ngram_size=ngram,
    )


def test_decoding_modes():
    assert resolve_decoding(decoding_args("raw_greedy")) == (
        "raw_greedy",
        False,
        1.0,
        None,
    )
    assert resolve_decoding(decoding_args("constrained_greedy")) == (
        "constrained_greedy",
        False,
        1.25,
        8,
    )
    assert resolve_decoding(decoding_args("sampled")) == (
        "sampled",
        True,
        1.0,
        None,
    )

    try:
        resolve_decoding(decoding_args("raw_greedy", penalty=1.25, ngram=8))
    except ValueError:
        pass
    else:
        raise AssertionError("raw_greedy accepted generation constraints")


def test_quality_metrics():
    assert longest_homopolymer_run("AAACCG") == 3
    assert longest_homopolymer_run("") == 0
    assert gc_difference_percent("AACC", "GGCC") == 50.0


def test_csv_diagnostics():
    record = PlastidRecord("TEST.1", "TEST.1 synthetic", "AAAACCCC")
    window = SlidingWindow(
        window_start=0,
        prompt_start=0,
        prompt_end=3,
        target_start=4,
        target_end=7,
        region="LSC",
        prompt="AAAA",
        true_suffix="CCCC",
        generated_suffix="CCCA",
        generated_length=4,
        accuracy_percent=75.0,
        decoding_mode="raw_greedy",
        fallback_count=0,
        longest_generated_run=3,
        n_count=0,
        gc_difference_percent=25.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = write_windows_csv(record, [window], directory)
        with open(path, newline="") as handle:
            row = next(csv.DictReader(handle))

    assert row["decoding_mode"] == "raw_greedy"
    assert row["fallback_count"] == "0"
    assert row["longest_generated_run"] == "3"
    assert row["n_count"] == "0"
    assert row["gc_difference_percent"] == "25.00"


if __name__ == "__main__":
    test_decoding_modes()
    test_quality_metrics()
    test_csv_diagnostics()
    print("All sliding evaluation tests passed!")
