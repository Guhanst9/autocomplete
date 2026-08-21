from analyze_generation_distance import calculate_distance_accuracy


def test_distance_accuracy_bins():
    rows = [
        {"generated_suffix": "ACGT", "true_suffix": "ACGA"},
        {"generated_suffix": "ACAA", "true_suffix": "ACGA"},
    ]
    positions, bins = calculate_distance_accuracy(rows, bin_size=2)

    assert [row["accuracy_percent"] for row in positions] == [100.0, 100.0, 50.0, 50.0]
    assert [row["accuracy_percent"] for row in bins] == [100.0, 50.0]
