from collections.abc import Iterable

from src.dna.prediction import TripletCodec


BASES = "ACGT"


def fit_frequency_counts(sequences: Iterable[str]) -> tuple[list[int], list[int]]:
    codec = TripletCodec()
    base_counts = [0] * len(BASES)
    triplet_counts = [0] * codec.vocab_size
    base_ids = {base: index for index, base in enumerate(BASES)}

    for sequence in sequences:
        sequence = sequence.upper()
        if set(sequence) - set(BASES):
            raise ValueError("baseline training sequences must contain only A/C/G/T")
        for base in sequence:
            base_counts[base_ids[base]] += 1
        for start in range(len(sequence) - 2):
            triplet_counts[codec.encode(sequence[start : start + 3])] += 1
    return base_counts, triplet_counts


def most_common_index(counts: list[int]) -> int:
    if not counts or sum(counts) == 0:
        raise ValueError("counts must contain at least one observation")
    return max(range(len(counts)), key=counts.__getitem__)


def most_common_base(counts: list[int]) -> str:
    if len(counts) != len(BASES):
        raise ValueError("base counts must have four entries")
    return BASES[most_common_index(counts)]


def most_common_triplet(counts: list[int], triplets: list[str] | None = None) -> str:
    codec = TripletCodec(triplets)
    if len(counts) != codec.vocab_size:
        raise ValueError("triplet counts must have 64 entries")
    return codec.decode(most_common_index(counts))


def generate_frequency(
    method: str,
    max_new_bases: int,
    base_counts: list[int],
    triplet_counts: list[int],
    triplets: list[str] | None = None,
) -> str:
    if max_new_bases < 0:
        raise ValueError("max_new_bases must be non-negative")
    if method == "most-common-base":
        return most_common_base(base_counts) * max_new_bases
    if method == "triplet-frequency":
        triplet = most_common_triplet(triplet_counts, triplets)
        repeats = (max_new_bases + 2) // 3
        return (triplet * repeats)[:max_new_bases]
    raise ValueError(f"unknown frequency baseline: {method}")
