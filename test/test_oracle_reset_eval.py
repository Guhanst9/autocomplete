import csv

from run_oracle_reset_eval import (
    distance_accuracy,
    load_rows,
    parse_intervals,
    summarize,
)


def test_load_rows_selects_evenly_spaced_windows(tmp_path):
    path = tmp_path / "windows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_start", "prompt", "true_suffix"],
        )
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {"window_start": index, "prompt": "ACGT", "true_suffix": "ACGT"}
            )

    rows = load_rows(path, 4)
    assert [int(row["window_start"]) for row in rows] == [0, 3, 6, 9]


def test_oracle_summary_and_distance_accuracy():
    rows = [
        {"true_suffix": "ACGT"},
        {"true_suffix": "AAAA"},
    ]
    generated = ["ACGA", "AAAA"]
    result = summarize(rows, generated, reset_interval=2)

    assert result["accuracy_percent"] == 87.5
    assert result["longest_run"] == 4
    assert distance_accuracy("ACGA", "ACGT", width=2) == (100.0, 50.0)
    assert parse_intervals("25, 50,100") == [25, 50, 100]
