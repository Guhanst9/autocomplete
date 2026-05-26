import gzip
import os
import random
import mmap
import struct
from typing import List, Tuple, Optional, Iterator

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
        
        # build vocab
        self.vocab = {self.pad_token: 0, self.mask_token: 1, self.unk_token: 2}
        for i, aa in enumerate(self.amino_acids):
            self.vocab[aa] = i + 3
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence: str) -> List[int]:
        """encode amino acid sequence to token ids"""
        tokens = []
        for aa in sequence.upper():
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                tokens.append(self.vocab[self.unk_token])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """decode token ids to amino acid sequence"""
        return ''.join([self.idx_to_token[idx] for idx in token_ids if idx != self.vocab[self.pad_token]])


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
        cache_dir: Optional[str] = None,
        max_sequences: Optional[int] = None,
    ):
        self.fasta_file = fasta_file
        self.l_max = l_max
        self.mask_prob = mask_prob
        self.max_sequences = max_sequences
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (input_ids, target_ids, attention_mask)"""
        sequence = self.sequences[idx].copy()
        seq_len = len(sequence)
        
        # pad or truncate to l_max
        if seq_len < self.l_max:
            padded = sequence + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.l_max - seq_len)
        else:
            padded = sequence[:self.l_max]
        
        # create input and target
        input_ids = padded.copy()
        target_ids = padded.copy()
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)
        
        # apply masking: 80% [MASK], 10% random, 10% unchanged
        num_mask = max(1, int(seq_len * self.mask_prob))
        mask_positions = random.sample(range(seq_len), min(num_mask, seq_len))
        valid_vocab = [i for i in range(self.tokenizer.vocab_size) if i != self.tokenizer.vocab[self.tokenizer.pad_token]]
        for pos in mask_positions:
            r = random.random()
            if r < 0.8:
                input_ids[pos] = self.tokenizer.vocab[self.tokenizer.mask_token]
            elif r < 0.9:
                input_ids[pos] = random.choice(valid_vocab)
            # else 10% unchanged
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


def create_dataloader(
    fasta_file: str,
    tokenizer: Optional[ProteinTokenizer] = None,
    l_max: int = 1024,
    mask_prob: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """create dataloader for protein sequences"""
    dataset = ProteinDataset(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        l_max=l_max,
        mask_prob=mask_prob,
        cache_dir=cache_dir,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
