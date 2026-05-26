import gzip
import os
import random
import mmap
import struct
from typing import List, Tuple, Optional, Iterator, Union

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm


class ProteinTokenizer:
    """amino acid tokenizer for protein sequences"""
    
    def __init__(self):
        # standard 20 amino acids
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
        # special tokens
        self.pad_token = '<PAD>'
        self.mask_token = '<MASK>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        
        # build vocab
        self.vocab = {self.pad_token: 0, self.mask_token: 1, self.unk_token: 2}
        for i, aa in enumerate(self.amino_acids):
            self.vocab[aa] = i + 3
        self.vocab[self.eos_token] = len(self.vocab)
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab[self.pad_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.eos_token_id = self.vocab[self.eos_token]
    
    def encode(self, sequence: str) -> List[int]:
        """encode amino acid sequence to token ids"""
        tokens = []
        for aa in sequence.upper():
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                tokens.append(self.vocab[self.unk_token])
        return tokens
    
    def decode(self, token_ids: List[int], stop_at_eos: bool = True) -> str:
        """decode token ids to amino acid sequence"""
        tokens = []
        for idx in token_ids:
            idx = int(idx)
            if idx == self.pad_token_id:
                continue
            if idx == self.eos_token_id and stop_at_eos:
                break
            if idx == self.eos_token_id:
                continue
            tokens.append(self.idx_to_token.get(idx, self.unk_token))
        return ''.join(tokens)


def _get_file_size(fasta_file: str) -> int:
    """Get file size for progress bar estimation."""
    if fasta_file.endswith(".gz"):
        # For gzipped files, estimate ~4x compression ratio
        return os.path.getsize(fasta_file) * 4
    return os.path.getsize(fasta_file)


def parse_fasta(fasta_file: str, show_progress: bool = True) -> List[Tuple[str, str]]:
    """Parse FASTA file (plain or gzipped .fasta.gz). Returns list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_sequence = []
    open_fn = gzip.open if fasta_file.endswith(".gz") else open
    mode = "rt" if fasta_file.endswith(".gz") else "r"
    
    file_size = _get_file_size(fasta_file)
    bytes_read = 0
    seq_count = 0

    print(f"📂 Loading FASTA file: {os.path.basename(fasta_file)}")
    
    with open_fn(fasta_file, mode) as f:
        pbar = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            desc="Parsing FASTA",
            disable=not show_progress
        ) if show_progress else None
        
        for line in f:
            if pbar:
                bytes_read += len(line.encode('utf-8'))
                pbar.update(len(line.encode('utf-8')))
            
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_sequence)))
                    seq_count += 1
                    if pbar and seq_count % 100000 == 0:
                        pbar.set_postfix({"sequences": f"{seq_count:,}"})
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_header is not None:
            sequences.append((current_header, "".join(current_sequence)))
        
        if pbar:
            pbar.close()
    
    print(f"✅ Parsed {len(sequences):,} sequences")
    return sequences


def stream_fasta(fasta_file: str, show_progress: bool = True) -> Iterator[Tuple[str, str]]:
    """Stream FASTA file one sequence at a time (memory efficient)."""
    open_fn = gzip.open if fasta_file.endswith(".gz") else open
    mode = "rt" if fasta_file.endswith(".gz") else "r"
    
    current_header = None
    current_sequence = []
    seq_count = 0
    
    with open_fn(fasta_file, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                if current_header is not None:
                    yield (current_header, "".join(current_sequence))
                    seq_count += 1
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header is not None:
            yield (current_header, "".join(current_sequence))


class ProteinDataset(Dataset):
    """Dataset for protein sequence autocompletion.
    
    Args:
        fasta_file: Path to FASTA file (plain or .gz)
        tokenizer: Optional tokenizer (creates default if None)
        l_max: Maximum sequence length
        mask_prob: Masking probability for MLM
        objective: "masked" for MLM or "autocomplete" for prefix-to-suffix LM
        prefix_length: Fixed prefix length for autocomplete, or None for random
        prefix_min_fraction: Minimum random prefix fraction for autocomplete
        prefix_max_fraction: Maximum random prefix fraction for autocomplete
        cache_dir: Directory to cache processed sequences
        max_sequences: Maximum number of sequences to load (None = all)
                      Use this to limit memory usage for large datasets
    """
    
    def __init__(
        self,
        fasta_file: str,
        tokenizer: Optional[ProteinTokenizer] = None,
        l_max: int = 1024,
        mask_prob: float = 0.15,
        objective: str = "masked",
        prefix_length: Optional[int] = None,
        prefix_min_fraction: float = 0.25,
        prefix_max_fraction: float = 0.70,
        cache_dir: Optional[str] = None,
        max_sequences: Optional[int] = None,
    ):
        self.fasta_file = fasta_file
        self.l_max = l_max
        self.mask_prob = mask_prob
        self.objective = objective
        self.prefix_length = prefix_length
        self.prefix_min_fraction = prefix_min_fraction
        self.prefix_max_fraction = prefix_max_fraction
        self.max_sequences = max_sequences

        if self.objective not in {"masked", "autocomplete"}:
            raise ValueError("objective must be 'masked' or 'autocomplete'")
        if not 0 < self.prefix_min_fraction <= self.prefix_max_fraction <= 1:
            raise ValueError("prefix fractions must satisfy 0 < min <= max <= 1")
        
        if tokenizer is None:
            self.tokenizer = ProteinTokenizer()
        else:
            self.tokenizer = tokenizer
        
        # load sequences (cache key includes basename and max_sequences)
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            base = os.path.basename(fasta_file).replace(".gz", "")
            suffix = f"_max{max_sequences}" if max_sequences else ""
            cache_file = os.path.join(cache_dir, f"sequences_{base}{suffix}_l{l_max}.pt")
        
        if cache_file and os.path.exists(cache_file):
            print(f"📦 Loading cached sequences from: {os.path.basename(cache_file)}")
            self.sequences = torch.load(cache_file)
            print(f"✅ Loaded {len(self.sequences):,} sequences from cache")
        else:
            self._load_sequences(fasta_file, l_max, max_sequences, cache_file)
    
    def _load_sequences(self, fasta_file: str, l_max: int, max_sequences: Optional[int], cache_file: Optional[str]):
        """Load sequences with memory-efficient streaming."""
        self.sequences = []
        filtered_count = 0
        
        limit_str = f", max={max_sequences:,}" if max_sequences else ""
        print(f"🔄 Loading sequences (l_max={l_max}{limit_str})...")
        print(f"📂 File: {os.path.basename(fasta_file)}")
        
        # Use streaming to avoid loading entire file into memory
        pbar = tqdm(
            stream_fasta(fasta_file, show_progress=False),
            desc="Processing sequences",
            unit="seq",
            total=max_sequences if max_sequences else None,
        )
        
        for header, seq in pbar:
            if max_sequences and len(self.sequences) >= max_sequences:
                pbar.set_postfix({"status": "limit reached"})
                break
            
            if len(seq) > 0 and len(seq) <= l_max:
                encoded = self.tokenizer.encode(seq)
                self.sequences.append(encoded)
                
                if len(self.sequences) % 500000 == 0:
                    pbar.set_postfix({
                        "kept": f"{len(self.sequences):,}",
                        "filtered": f"{filtered_count:,}"
                    })
            elif len(seq) > l_max:
                filtered_count += 1
        
        pbar.close()
        
        print(f"✅ Loaded {len(self.sequences):,} sequences")
        if filtered_count > 0:
            print(f"⚠️  Filtered out {filtered_count:,} sequences (longer than {l_max})")
        
        # Estimate memory usage
        mem_mb = sum(len(s) * 8 for s in self.sequences) / 1024 / 1024
        print(f"💾 Estimated memory: ~{mem_mb:.1f} MB")
        
        if cache_file:
            print(f"💾 Saving cache to: {os.path.basename(cache_file)}")
            torch.save(self.sequences, cache_file)
            print(f"✅ Cache saved!")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """return tensors for the requested training objective"""
        if self.objective == "autocomplete":
            return self._getitem_autocomplete(idx)
        return self._getitem_masked(idx)

    def _pad(self, tokens: List[int]) -> List[int]:
        if len(tokens) < self.l_max:
            return tokens + [self.tokenizer.pad_token_id] * (self.l_max - len(tokens))
        return tokens[:self.l_max]

    def _sample_prefix_length(self, amino_len: int) -> int:
        if self.prefix_length is not None:
            return min(max(1, self.prefix_length), amino_len)

        min_prefix = max(1, int(round(amino_len * self.prefix_min_fraction)))
        max_prefix = max(min_prefix, int(round(amino_len * self.prefix_max_fraction)))
        max_prefix = min(max_prefix, amino_len)
        return random.randint(min_prefix, max_prefix)

    def _getitem_masked(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (input_ids, target_ids, attention_mask) for masked LM"""
        sequence = self.sequences[idx].copy()
        seq_len = len(sequence)
        
        # pad or truncate to l_max
        padded = self._pad(sequence)
        seq_len = min(seq_len, self.l_max)
        
        # create input and target
        input_ids = padded.copy()
        target_ids = padded.copy()
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)
        
        # apply masking: 80% [MASK], 10% random, 10% unchanged
        num_mask = max(1, int(seq_len * self.mask_prob))
        mask_positions = random.sample(range(seq_len), min(num_mask, seq_len))
        valid_vocab = [
            i for i in range(self.tokenizer.vocab_size)
            if i not in {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        ]
        for pos in mask_positions:
            r = random.random()
            if r < 0.8:
                input_ids[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                input_ids[pos] = random.choice(valid_vocab)
            # else 10% unchanged
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def _getitem_autocomplete(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (input_ids, next_token_targets, attention_mask, loss_mask)."""
        sequence = self.sequences[idx].copy()
        if self.l_max < 2:
            raise ValueError("l_max must be at least 2 for autocomplete training")

        # Reserve one slot for EOS so the model can learn when to stop.
        sequence = sequence[: self.l_max - 1]
        amino_len = max(1, len(sequence))
        tokens = sequence + [self.tokenizer.eos_token_id]
        seq_len = len(tokens)
        prefix_len = self._sample_prefix_length(amino_len)

        input_ids = self._pad(tokens)
        target_ids = self._pad(tokens[1:] + [self.tokenizer.pad_token_id])
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)

        # Position i predicts token i+1, so prefix_len amino acids make the
        # first graded suffix prediction at position prefix_len - 1.
        loss_mask = [0] * self.l_max
        start = max(0, prefix_len - 1)
        for pos in range(start, seq_len - 1):
            loss_mask[pos] = 1

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.long),
        )


def create_dataloader(
    fasta_file: str,
    tokenizer: Optional[ProteinTokenizer] = None,
    l_max: int = 1024,
    mask_prob: float = 0.15,
    objective: str = "masked",
    prefix_length: Optional[int] = None,
    prefix_min_fraction: float = 0.25,
    prefix_max_fraction: float = 0.70,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    max_sequences: Optional[int] = None,
) -> DataLoader:
    """create dataloader for protein sequences"""
    dataset = ProteinDataset(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        l_max=l_max,
        mask_prob=mask_prob,
        objective=objective,
        prefix_length=prefix_length,
        prefix_min_fraction=prefix_min_fraction,
        prefix_max_fraction=prefix_max_fraction,
        cache_dir=cache_dir,
        max_sequences=max_sequences,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
