import gzip
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset

from src.dna.prediction import TripletCodec, normalize_prediction_unit


class DnaTokenizer:
    def __init__(self, include_n: bool = False, vocab: Optional[dict[str, int]] = None):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.eos_token = "<EOS>"
        if vocab is not None:
            self.vocab = dict(vocab)
            self.bases = [base for base in ["A", "C", "G", "T", "N"] if base in self.vocab]
        else:
            self.bases = ["A", "C", "G", "T"] + (["N"] if include_n else [])
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.eos_token: 2,
            }
            for base in self.bases:
                self.vocab[base] = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.eos_token_id = self.vocab[self.eos_token]

    def normalize(self, sequence: str) -> str:
        sequence = sequence.upper().replace("U", "T")
        allowed = set(self.bases)
        return "".join(base if base in allowed else self.unk_token for base in sequence)

    def encode(self, sequence: str) -> list[int]:
        normalized = self.normalize(sequence)
        tokens = []
        i = 0
        while i < len(normalized):
            if normalized.startswith(self.unk_token, i):
                tokens.append(self.unk_token_id)
                i += len(self.unk_token)
            else:
                tokens.append(self.vocab.get(normalized[i], self.unk_token_id))
                i += 1
        return tokens

    def decode(self, token_ids: Iterable[int], stop_at_eos: bool = True) -> str:
        out = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.pad_token_id:
                continue
            if token_id == self.eos_token_id and stop_at_eos:
                break
            if token_id == self.eos_token_id:
                continue
            token = self.idx_to_token.get(token_id, "N")
            out.append(token if token in self.bases else "N")
        return "".join(out)


def stream_fasta(path: str) -> Iterable[tuple[str, str]]:
    open_fn = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    header = None
    chunks = []
    with open_fn(path, mode) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def accession_from_header(header: str) -> str:
    return header.split()[0] if header else ""


def load_dna_records(
    fasta_file: str,
    tokenizer: DnaTokenizer,
    max_records: Optional[int],
    exclude_accessions: set[str],
) -> list[tuple[str, str]]:
    records = []
    for header, sequence in stream_fasta(fasta_file):
        accession = accession_from_header(header)
        if accession in exclude_accessions:
            continue
        normalized = tokenizer.normalize(sequence)
        if not normalized or tokenizer.unk_token in normalized:
            continue
        records.append((header, normalized))
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise ValueError(f"No usable DNA records were loaded from {fasta_file}")
    return records


def split_records(
    records: list[tuple[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(records) < 2:
        raise ValueError("record-level train/val split needs at least 2 records")
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


@dataclass
class Window:
    tokens: list[int]
    ends_sequence: bool


class DnaWindowDataset(Dataset):
    def __init__(
        self,
        fasta_file: str,
        tokenizer: DnaTokenizer,
        l_max: int,
        stride: Optional[int],
        max_windows: Optional[int],
        windows_per_record: Optional[int],
        prefix_min_fraction: float,
        prefix_max_fraction: float,
        seed: int,
        records: list[tuple[str, str]],
        prediction_unit: str = "base",
        triplet_codec: Optional[TripletCodec] = None,
    ):
        if l_max < 4:
            raise ValueError("l_max must be at least 4")
        if not 0 < prefix_min_fraction <= prefix_max_fraction < 1:
            raise ValueError("prefix fractions must satisfy 0 < min <= max < 1")

        self.fasta_file = fasta_file
        self.tokenizer = tokenizer
        self.l_max = l_max
        self.stride = stride or l_max
        self.prefix_min_fraction = prefix_min_fraction
        self.prefix_max_fraction = prefix_max_fraction
        self.prediction_unit = normalize_prediction_unit(prediction_unit)
        self.triplet_codec = triplet_codec or TripletCodec()
        self.windows: list[Window] = []
        self.first_header = ""
        self.first_sequence = ""
        rng = random.Random(seed)

        records_seen = 0
        for header, normalized in records:
            if not normalized:
                continue
            if not self.first_sequence:
                self.first_header = header
                self.first_sequence = normalized
            encoded = tokenizer.encode(normalized)
            if windows_per_record is None:
                self._add_windows(encoded, max_windows)
            else:
                self._sample_windows(encoded, windows_per_record, rng, max_windows)
            records_seen += 1
            if records_seen % 1000 == 0:
                print(
                    f"  Loaded {records_seen:,} records, collected {len(self.windows):,} windows...",
                    flush=True,
                )
            if max_windows is not None and len(self.windows) >= max_windows:
                break

        if not self.windows:
            raise ValueError(f"No DNA windows were loaded from {fasta_file}")

        rng.shuffle(self.windows)

    def _add_windows(self, encoded: list[int], max_windows: Optional[int]) -> None:
        internal_window_len = self.l_max
        final_window_len = self.l_max - 1 if self.prediction_unit == "base" else self.l_max
        minimum_length = 3 if self.prediction_unit == "base" else 4
        for start in range(0, len(encoded), self.stride):
            remaining = len(encoded) - start
            reaches_end = remaining <= final_window_len
            chunk_len = final_window_len if reaches_end else internal_window_len
            chunk = encoded[start : start + chunk_len]
            if len(chunk) < minimum_length:
                continue
            ends_sequence = start + len(chunk) >= len(encoded)
            self.windows.append(Window(tokens=chunk, ends_sequence=ends_sequence))
            if max_windows is not None and len(self.windows) >= max_windows:
                return

    def _sample_windows(
        self,
        encoded: list[int],
        windows_per_record: int,
        rng: random.Random,
        max_windows: Optional[int],
    ) -> None:
        if windows_per_record <= 0:
            raise ValueError("windows_per_record must be positive")
        starts = list(range(0, len(encoded), self.stride))
        rng.shuffle(starts)
        final_window_len = self.l_max - 1 if self.prediction_unit == "base" else self.l_max
        minimum_length = 3 if self.prediction_unit == "base" else 4
        for start in starts[:windows_per_record]:
            remaining = len(encoded) - start
            reaches_end = remaining <= final_window_len
            chunk_len = final_window_len if reaches_end else self.l_max
            chunk = encoded[start : start + chunk_len]
            if len(chunk) < minimum_length:
                continue
            ends_sequence = start + len(chunk) >= len(encoded)
            self.windows.append(Window(tokens=chunk, ends_sequence=ends_sequence))
            if max_windows is not None and len(self.windows) >= max_windows:
                return

    def __len__(self) -> int:
        return len(self.windows)

    def window_tokens(self, idx: int) -> list[int]:
        item = self.windows[idx]
        tokens = item.tokens.copy()
        if self.prediction_unit == "base" and item.ends_sequence and len(tokens) < self.l_max:
            tokens.append(self.tokenizer.eos_token_id)
        return tokens[: self.l_max]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.window_tokens(idx)
        seq_len = len(tokens)

        min_prefix = max(1, int(round(seq_len * self.prefix_min_fraction)))
        max_prefix = max(min_prefix, int(round(seq_len * self.prefix_max_fraction)))
        max_prefix = min(max_prefix, seq_len - 1)
        prefix_len = random.randint(min_prefix, max_prefix)

        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.l_max - seq_len)
        if self.prediction_unit == "base":
            target_ids = tokens[1:] + [self.tokenizer.pad_token_id]
            target_ids += [self.tokenizer.pad_token_id] * (self.l_max - len(target_ids))
            last_target_position = seq_len - 1
        else:
            target_ids = [0] * self.l_max
            for pos in range(max(0, seq_len - 3)):
                triplet = self.tokenizer.decode(tokens[pos + 1 : pos + 4], stop_at_eos=False)
                if len(triplet) == 3 and set(triplet) <= set("ACGT"):
                    target_ids[pos] = self.triplet_codec.encode(triplet)
            last_target_position = seq_len - 3
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)

        loss_mask = [0] * self.l_max
        start = max(0, prefix_len - 1)
        for pos in range(start, max(start, last_target_position)):
            loss_mask[pos] = 1

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.long),
        )
