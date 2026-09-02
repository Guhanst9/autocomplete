from collections.abc import Iterable

import torch

from src.baselines.frequency import most_common_index
from src.dna.prediction import TripletCodec


BASE_TO_ID = {base: index for index, base in enumerate("ACGT")}
SUPPORTED_ORDERS = (1, 3, 6)


def _sequence_codes(sequence: str) -> torch.Tensor:
    lookup = torch.full((256,), -1, dtype=torch.int64)
    for base, index in BASE_TO_ID.items():
        lookup[ord(base)] = index
    byte_values = torch.frombuffer(bytearray(sequence, "ascii"), dtype=torch.uint8)
    codes = lookup[byte_values.long()]
    if bool((codes < 0).any()):
        raise ValueError("baseline training sequences must contain only A/C/G/T")
    return codes


def fit_count_baselines(
    sequences: Iterable[str],
) -> tuple[list[int], list[int], dict[str, dict[str, list[int]]]]:
    base_counts = torch.zeros(4, dtype=torch.int64)
    triplet_counts = torch.zeros(64, dtype=torch.int64)
    order_six = torch.zeros((4**6, 64), dtype=torch.int64)
    prefix_rows: list[tuple[str, str]] = []

    for sequence in sequences:
        sequence = sequence.upper()
        codes = _sequence_codes(sequence)
        base_counts += torch.bincount(codes, minlength=4)
        if len(codes) >= 3:
            targets = codes[:-2] * 16 + codes[1:-1] * 4 + codes[2:]
            triplet_counts += torch.bincount(targets, minlength=64)
        if len(codes) >= 9:
            context = torch.zeros(len(codes) - 8, dtype=torch.int64)
            for offset in range(6):
                context = context * 4 + codes[offset : offset + len(context)]
            target = (
                codes[6:-2] * 16
                + codes[7:-1] * 4
                + codes[8:]
            )
            joint = torch.bincount(context * 64 + target, minlength=4**6 * 64)
            order_six += joint.reshape(4**6, 64)
        for target_start in range(min(6, max(0, len(sequence) - 2))):
            if target_start:
                prefix_rows.append(
                    (sequence[:target_start], sequence[target_start : target_start + 3])
                )

    tables: dict[str, dict[str, list[int]]] = {str(order): {} for order in range(1, 7)}
    codec = TripletCodec()
    for order in range(1, 7):
        reshaped = order_six.reshape(4 ** (6 - order), 4**order, 64).sum(dim=0)
        table = tables[str(order)]
        for context_id, row in enumerate(reshaped.tolist()):
            if sum(row):
                table[_decode_context(context_id, order)] = row

    for history, target in prefix_rows:
        target_id = codec.encode(target)
        for order in range(1, min(6, len(history)) + 1):
            context = history[-order:]
            row = tables[str(order)].setdefault(context, [0] * 64)
            row[target_id] += 1
    return base_counts.tolist(), triplet_counts.tolist(), tables


def fit_markov_counts(sequences: Iterable[str]) -> dict[str, dict[str, list[int]]]:
    return fit_count_baselines(sequences)[2]


def _decode_context(value: int, length: int) -> str:
    bases = "ACGT"
    output = ["A"] * length
    for index in range(length - 1, -1, -1):
        output[index] = bases[value % 4]
        value //= 4
    return "".join(output)


def predict_triplet(
    history: str,
    order: int,
    markov_counts: dict[str, dict[str, list[int]]],
    global_triplet_counts: list[int],
    triplets: list[str] | None = None,
) -> str:
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"Markov order must be one of {SUPPORTED_ORDERS}")
    codec = TripletCodec(triplets)
    for context_length in range(min(order, len(history)), 0, -1):
        row = markov_counts.get(str(context_length), {}).get(history[-context_length:])
        if row and sum(row):
            return codec.decode(most_common_index(row))
    return codec.decode(most_common_index(global_triplet_counts))


def generate_markov(
    prompt: str,
    max_new_bases: int,
    order: int,
    markov_counts: dict[str, dict[str, list[int]]],
    global_triplet_counts: list[int],
    triplets: list[str] | None = None,
) -> str:
    if max_new_bases < 0:
        raise ValueError("max_new_bases must be non-negative")
    history = prompt
    generated = ""
    while len(generated) < max_new_bases:
        triplet = predict_triplet(
            history,
            order,
            markov_counts,
            global_triplet_counts,
            triplets,
        )
        generated += triplet
        history += triplet
    return generated[:max_new_bases]
