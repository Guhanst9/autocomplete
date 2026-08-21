import csv

from run_decoding_sweep import aggregate, evenly_spaced_starts, summarize_csv


def test_sweep_summary(tmp_path):
    path = tmp_path / "windows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generated_suffix", "true_suffix"],
        )
        writer.writeheader()
        writer.writerow({"generated_suffix": "ACGT", "true_suffix": "ACGA"})
        writer.writerow({"generated_suffix": "AAAA", "true_suffix": "AAAA"})

    result = summarize_csv(path, temperature=0.8, seed=13)
    assert result["rows"] == 2
    assert result["accuracy_percent"] == 87.5
    assert result["longest_run"] == 4
    assert result["rows_with_runs_over_20"] == 0

    combined = aggregate([result, {**result, "seed": 29, "accuracy_percent": 62.5}])
    assert combined[0]["seeds"] == 2
    assert combined[0]["mean_accuracy_percent"] == 75.0


def test_evenly_spaced_starts_cover_full_range():
    assert evenly_spaced_starts(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced_starts([0, 2], 4) == [0, 2]
