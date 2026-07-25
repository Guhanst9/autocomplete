import csv
import tempfile

from src.sliding_eval.fasta import PlastidRecord
from src.sliding_eval.generation import gc_difference_percent, longest_homopolymer_run
from src.sliding_eval.windows import SlidingWindow, write_windows_csv


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
        longest_generated_run=3,
        n_count=0,
        gc_difference_percent=25.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = write_windows_csv(record, [window], directory)
        with open(path, newline="") as handle:
            row = next(csv.DictReader(handle))

    assert row["decoding_mode"] == "raw_greedy"
    assert row["longest_generated_run"] == "3"
    assert row["n_count"] == "0"
    assert row["gc_difference_percent"] == "25.00"


if __name__ == "__main__":
    test_quality_metrics()
    test_csv_diagnostics()
    print("All sliding evaluation tests passed!")
