from src.dna.data import DnaTokenizer, DnaWindowDataset


def make_dataset(seed: int) -> DnaWindowDataset:
    sequence = "".join("ACGT"[(index * index + index // 7) % 4] for index in range(4096))
    return DnaWindowDataset(
        fasta_file="memory.fasta",
        tokenizer=DnaTokenizer(),
        l_max=64,
        stride=16,
        max_windows=4,
        windows_per_record=4,
        prefix_min_fraction=0.25,
        prefix_max_fraction=0.70,
        seed=seed,
        records=[("record", sequence)],
    )


def test_resampling_is_repeatable_and_changes_windows():
    dataset = make_dataset(13)
    first = [window.tokens for window in dataset.windows]

    dataset.resample(14)
    second = [window.tokens for window in dataset.windows]
    dataset.resample(13)
    repeated = [window.tokens for window in dataset.windows]

    assert first == repeated
    assert first != second


def test_resampling_replaces_too_short_end_fragment():
    for seed in range(20):
        dataset = DnaWindowDataset(
            fasta_file="memory.fasta",
            tokenizer=DnaTokenizer(),
            l_max=64,
            stride=64,
            max_windows=None,
            windows_per_record=2,
            prefix_min_fraction=0.25,
            prefix_max_fraction=0.70,
            seed=seed,
            records=[("record", "A" * 130)],
            prediction_unit="triplet",
        )
        assert len(dataset) == 2


def test_legacy_resampling_can_reproduce_historical_window_fingerprints():
    counts = []
    for seed in range(20):
        dataset = DnaWindowDataset(
            fasta_file="memory.fasta",
            tokenizer=DnaTokenizer(),
            l_max=64,
            stride=64,
            max_windows=None,
            windows_per_record=2,
            prefix_min_fraction=0.25,
            prefix_max_fraction=0.70,
            seed=seed,
            records=[("record", "A" * 130)],
            prediction_unit="triplet",
            filter_short_window_starts=False,
        )
        counts.append(len(dataset))

    assert 1 in counts
    assert 2 in counts
