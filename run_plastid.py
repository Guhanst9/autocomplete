import argparse
import gzip
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from src.models.s4_model import S4ProteinModel
except ImportError:
    from models.s4_model import S4ProteinModel


class PlastidTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.eos_token = "<EOS>"
        self.bases = ["A", "C", "G", "T", "N"]
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
        return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in sequence)

    def encode(self, sequence: str) -> list[int]:
        return [self.vocab.get(base, self.unk_token_id) for base in self.normalize(sequence)]

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


@dataclass
class Window:
    tokens: list[int]
    ends_sequence: bool


class PlastidAutocompleteDataset(Dataset):
    def __init__(
        self,
        fasta_file: str,
        tokenizer: PlastidTokenizer,
        l_max: int = 256,
        stride: Optional[int] = None,
        max_records: Optional[int] = 8,
        max_windows: Optional[int] = 512,
        windows_per_record: Optional[int] = None,
        prefix_min_fraction: float = 0.25,
        prefix_max_fraction: float = 0.70,
        seed: int = 13,
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
        self.windows: list[Window] = []
        self.first_header = ""
        self.first_sequence = ""
        rng = random.Random(seed)

        records_seen = 0
        for header, sequence in stream_fasta(fasta_file):
            if max_records is not None and records_seen >= max_records:
                break
            normalized = tokenizer.normalize(sequence)
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
                    f"  Loaded {records_seen:,} records, sampled {len(self.windows):,} windows...",
                    flush=True,
                )
            if max_windows is not None and len(self.windows) >= max_windows:
                break

        if not self.windows:
            raise ValueError(f"No DNA windows were loaded from {fasta_file}")

        rng.shuffle(self.windows)

    def _add_windows(self, encoded: list[int], max_windows: Optional[int]) -> None:
        # reserve eos space only for true record endings
        internal_window_len = self.l_max
        final_window_len = self.l_max - 1
        for start in range(0, len(encoded), self.stride):
            remaining = len(encoded) - start
            reaches_end = remaining <= final_window_len
            chunk_len = final_window_len if reaches_end else internal_window_len
            chunk = encoded[start : start + chunk_len]
            if len(chunk) < 3:
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
        for start in starts[:windows_per_record]:
            remaining = len(encoded) - start
            reaches_end = remaining <= self.l_max - 1
            chunk_len = self.l_max - 1 if reaches_end else self.l_max
            chunk = encoded[start : start + chunk_len]
            if len(chunk) < 3:
                continue
            ends_sequence = start + len(chunk) >= len(encoded)
            self.windows.append(Window(tokens=chunk, ends_sequence=ends_sequence))
            if max_windows is not None and len(self.windows) >= max_windows:
                return

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.windows[idx]
        tokens = item.tokens.copy()
        if item.ends_sequence and len(tokens) < self.l_max:
            tokens.append(self.tokenizer.eos_token_id)
        tokens = tokens[: self.l_max]
        seq_len = len(tokens)

        min_prefix = max(1, int(round(seq_len * self.prefix_min_fraction)))
        max_prefix = max(min_prefix, int(round(seq_len * self.prefix_max_fraction)))
        max_prefix = min(max_prefix, seq_len - 1)
        prefix_len = random.randint(min_prefix, max_prefix)

        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.l_max - seq_len)
        target_ids = tokens[1:] + [self.tokenizer.pad_token_id]
        target_ids += [self.tokenizer.pad_token_id] * (self.l_max - len(target_ids))
        attention_mask = [1] * seq_len + [0] * (self.l_max - seq_len)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run S4 autocomplete on plastid DNA data.")
    parser.add_argument("--fasta_file", type=str, default="data/plastid/plant_chloroplast_refseq_500.fna.gz")
    parser.add_argument("--output_dir", type=str, default="outputs/plastid_quick")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to continue training from.")
    parser.add_argument("--l_max", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_records", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--windows_per_record", type=int, default=None,
                        help="Sample this many windows from each FASTA record before applying max_windows.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel_type", type=str, default="diag", choices=["diag", "nplr"])
    parser.add_argument("--prefix_min_fraction", type=float, default=0.25)
    parser.add_argument("--prefix_max_fraction", type=float, default=0.70)
    parser.add_argument("--prompt_length", type=int, default=80)
    parser.add_argument("--generate_length", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for the final sample.")
    parser.add_argument("--eval_train", action="store_true", help="Also report teacher-forced metrics on train windows.")
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unpack_batch(batch, device):
    input_ids, target_ids, attention_mask, loss_mask = batch
    return (
        input_ids.to(device),
        target_ids.to(device),
        attention_mask.to(device),
        loss_mask.to(device),
    )


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train"):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(
            input_ids,
            target_ids,
            attention_mask,
            loss_mask=loss_mask,
            objective="autocomplete",
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    for batch in tqdm(loader, desc="Val"):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)
        logits = model(input_ids, attention_mask=attention_mask)
        loss = model.compute_loss(
            input_ids,
            target_ids,
            attention_mask,
            loss_mask=loss_mask,
            objective="autocomplete",
        )
        total_loss += loss.item()
        score_mask = loss_mask == 1
        targets = target_ids[score_mask]
        scored_logits = logits[score_mask]
        if targets.numel() == 0:
            continue
        total += targets.numel()
        correct_top1 += (scored_logits.argmax(dim=-1) == targets).sum().item()
        top3 = scored_logits.topk(min(3, scored_logits.shape[-1]), dim=-1).indices
        correct_top3 += (top3 == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    avg_loss = total_loss / max(1, len(loader))
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "top1": correct_top1 / total if total else 0.0,
        "top3": correct_top3 / total if total else 0.0,
        "tokens": total,
    }


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, args, device):
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.generate_length,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=True,
        forbidden_token_ids=(tokenizer.pad_token_id, tokenizer.unk_token_id),
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=min(20, args.generate_length),
    )
    return tokenizer.decode(out[0].tolist(), stop_at_eos=True)


def continuation_accuracy(prompt: str, generated: str, reference: str) -> Optional[float]:
    generated_suffix = generated[len(prompt):]
    if not generated_suffix:
        return None
    reference_suffix = reference[len(prompt) : len(prompt) + len(generated_suffix)]
    if len(reference_suffix) != len(generated_suffix):
        return None
    matches = sum(a == b for a, b in zip(generated_suffix, reference_suffix))
    return matches / len(generated_suffix)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    tokenizer = PlastidTokenizer()

    print("Loading plastid FASTA windows...", flush=True)
    dataset = PlastidAutocompleteDataset(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        l_max=args.l_max,
        stride=args.stride,
        max_records=args.max_records,
        max_windows=args.max_windows,
        windows_per_record=args.windows_per_record,
        prefix_min_fraction=args.prefix_min_fraction,
        prefix_max_fraction=args.prefix_max_fraction,
        seed=args.seed,
    )
    val_size = max(1, int(round(len(dataset) * 0.1)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = S4ProteinModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=args.dropout,
        kernel_type=args.kernel_type,
        bidirectional=False,
        l_max=args.l_max,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.l_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val_loss", best_val)

    print("Plastid S4 run")
    print(f"  Device: {device}")
    print(f"  FASTA: {args.fasta_file}")
    print(f"  Records loaded: up to {args.max_records}")
    print(f"  Windows: {len(dataset)}")
    print(f"  Train/val windows: {train_size}/{val_size}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model: d_model={args.d_model}, d_state={args.d_state}, n_layers={args.n_layers}")
    if args.resume:
        print(f"  Resume: {args.resume} starting at epoch {start_epoch}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        train_metrics = evaluate(model, train_loader, device) if args.eval_train else None
        metrics = evaluate(model, val_loader, device)
        is_best = metrics["loss"] < best_val
        best_val = min(best_val, metrics["loss"])
        save = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "objective": "autocomplete",
            "data_type": "plastid_dna",
            "best_val_loss": best_val,
            "tokenizer_vocab": tokenizer.vocab,
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": model.d_model,
                "d_state": model.d_state,
                "n_layers": model.n_layers,
                "kernel_type": model.kernel_type,
                "eos_token_id": tokenizer.eos_token_id,
                "l_max": args.l_max,
            },
        }
        torch.save(save, os.path.join(args.output_dir, "last.pt"))
        if is_best or not os.path.exists(os.path.join(args.output_dir, "best.pt")):
            torch.save(save, os.path.join(args.output_dir, "best.pt"))
        line = (
            f"Epoch {epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_ppl={metrics['perplexity']:.3f} "
            f"top1={metrics['top1']:.4f} "
            f"top3={metrics['top3']:.4f}"
        )
        if train_metrics is not None:
            line += (
                f" train_top1={train_metrics['top1']:.4f}"
                f" train_top3={train_metrics['top3']:.4f}"
            )
        print(line)

    prompt = dataset.first_sequence[: args.prompt_length]
    generated = generate_sample(model, tokenizer, prompt, args, device)
    exact_acc = continuation_accuracy(prompt, generated, dataset.first_sequence)
    print()
    print("Prompt header:", dataset.first_header)
    print("Prompt:", prompt)
    print("Generated:", generated)
    if exact_acc is not None:
        print(f"Exact continuation accuracy: {exact_acc:.4f}")
    print("Saved:", os.path.join(args.output_dir, "best.pt"))


if __name__ == "__main__":
    main()
