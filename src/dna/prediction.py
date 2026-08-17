from itertools import product
from typing import Iterable

import torch


class TripletCodec:
    def __init__(self, triplets: Iterable[str] | None = None):
        values = list(triplets) if triplets is not None else [
            "".join(parts) for parts in product("ACGT", repeat=3)
        ]
        if len(values) != 64 or len(set(values)) != 64:
            raise ValueError("triplet vocabulary must contain 64 unique triplets")
        if any(len(value) != 3 or set(value) - set("ACGT") for value in values):
            raise ValueError("triplet vocabulary may contain only three-base A/C/G/T values")
        self.triplets = values
        self.vocab = {triplet: index for index, triplet in enumerate(values)}

    @property
    def vocab_size(self) -> int:
        return len(self.triplets)

    def encode(self, triplet: str) -> int:
        return self.vocab[triplet]

    def decode(self, class_id: int) -> str:
        return self.triplets[int(class_id)]

    def base_ids(self, tokenizer, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [[tokenizer.vocab[base] for base in triplet] for triplet in self.triplets],
            dtype=torch.long,
            device=device,
        )


def normalize_prediction_unit(value: str | None) -> str:
    unit = value or "base"
    if unit not in {"base", "triplet"}:
        raise ValueError("prediction_unit must be 'base' or 'triplet'")
    return unit
