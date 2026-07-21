import argparse
import gzip
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from src.models.s4_model import S4ProteinModel
except ImportError:
    from models.s4_model import S4ProteinModel


class PlastidTokenizer:
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


def tokenizer_from_checkpoint(checkpoint: dict) -> PlastidTokenizer:
    vocab = checkpoint.get("tokenizer_vocab")
    if vocab is None:
        return PlastidTokenizer(include_n=True)
    return PlastidTokenizer(vocab=vocab)


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


def load_plastid_records(
    fasta_file: str,
    tokenizer: PlastidTokenizer,
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
        raise ValueError(f"No usable plastid records were loaded from {fasta_file}")
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
        records: Optional[list[tuple[str, str]]] = None,
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

        source_records = records
        if source_records is None:
            source_records = load_plastid_records(
                fasta_file=fasta_file,
                tokenizer=tokenizer,
                max_records=max_records,
                exclude_accessions=set(),
            )

        records_seen = 0
        for header, normalized in source_records:
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

    def window_tokens(self, idx: int) -> list[int]:
        item = self.windows[idx]
        tokens = item.tokens.copy()
        if item.ends_sequence and len(tokens) < self.l_max:
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
    parser.add_argument("--fasta_file", type=str, default="data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz")
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
    parser.add_argument("--kernel_type", type=str, default="diag", choices=["diag"])
    parser.add_argument("--model_variant", type=str, default="s4d_v2", choices=["legacy", "s4d_v2"])
    parser.add_argument("--ssm_lr", type=float, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--exclude_accession", action="append", default=["NC_053550.1"])
    parser.add_argument("--prefix_min_fraction", type=float, default=0.25)
    parser.add_argument("--prefix_max_fraction", type=float, default=0.70)
    parser.add_argument("--prompt_length", type=int, default=80)
    parser.add_argument("--generate_length", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for the final sample.")
    parser.add_argument("--eval_train", action="store_true", help="Also report teacher-forced metrics on train windows.")
    parser.add_argument("--rollout_length", type=int, default=64)
    parser.add_argument("--rollout_loss_weight", type=float, default=0.1)
    parser.add_argument("--scheduled_sampling_max", type=float, default=0.20)
    parser.add_argument("--free_eval_windows", type=int, default=8)
    parser.add_argument("--free_eval_prompt_length", type=int, default=512)
    parser.add_argument("--free_eval_generate_length", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(
    model: S4ProteinModel,
    lr: float,
    weight_decay: float,
    ssm_lr: Optional[float] = None,
) -> torch.optim.AdamW:
    if model.model_variant == "legacy":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ssm_suffixes = (
        "kernel.log_dt",
        "kernel.log_A_real",
        "kernel.A_imag",
    )
    ssm_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if name.endswith(ssm_suffixes):
            ssm_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    if not ssm_parameters:
        raise ValueError("s4d_v2 did not expose trainable SSM parameters")

    return torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": lr, "weight_decay": weight_decay},
            {
                "params": ssm_parameters,
                "lr": lr if ssm_lr is None else ssm_lr,
                "weight_decay": 0.0,
            },
        ]
    )


def unpack_batch(batch, device):
    input_ids, target_ids, attention_mask, loss_mask = batch
    return (
        input_ids.to(device),
        target_ids.to(device),
        attention_mask.to(device),
        loss_mask.to(device),
    )


def scheduled_sampling_for_epoch(epoch_offset: int, total_epochs: int, max_probability: float) -> float:
    if total_epochs <= 1:
        return 0.0
    return max_probability * epoch_offset / max(1, total_epochs - 1)


def recurrent_rollout_loss(
    model: S4ProteinModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rollout_length: int,
    scheduled_sampling_probability: float,
) -> torch.Tensor:
    if rollout_length <= 0:
        return input_ids.new_tensor(0.0, dtype=torch.float)

    losses = []
    batch_size = input_ids.size(0)
    for batch_idx in range(batch_size):
        seq_len = int(attention_mask[batch_idx].sum().item())
        if seq_len < 3:
            continue

        sequence = input_ids[batch_idx, :seq_len]
        prompt_len = max(1, seq_len - rollout_length)
        targets = sequence[prompt_len:]
        if targets.numel() == 0:
            continue

        states = [block.default_state(1, input_ids.device) for block in model.blocks]
        for token in sequence[:prompt_len]:
            x = model.embed(token.view(1))
            for block_idx, block in enumerate(model.blocks):
                x, states[block_idx] = block.step(x, states[block_idx])

        for target in targets:
            logits = model.lm_head(model.ln_f(x))
            losses.append(
                torch.nn.functional.cross_entropy(
                    logits,
                    target.view(1),
                    reduction="mean",
                )
            )
            predicted = logits.argmax(dim=-1).detach()
            use_prediction = (
                scheduled_sampling_probability > 0
                and torch.rand((), device=input_ids.device).item() < scheduled_sampling_probability
            )
            next_token = predicted if use_prediction else target.view(1)
            x = model.embed(next_token)
            for block_idx, block in enumerate(model.blocks):
                x, states[block_idx] = block.step(x, states[block_idx])

    if not losses:
        return input_ids.new_tensor(0.0, dtype=torch.float)
    return torch.stack(losses).mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip,
    rollout_length,
    rollout_loss_weight,
    scheduled_sampling_probability,
):
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
        if rollout_loss_weight > 0 and rollout_length > 0:
            rollout_loss = recurrent_rollout_loss(
                model,
                input_ids,
                attention_mask,
                rollout_length,
                scheduled_sampling_probability,
            )
            loss = loss + rollout_loss_weight * rollout_loss
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


def longest_run(sequence: str) -> int:
    best = 0
    current = 0
    previous = None
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        best = max(best, current)
    return best


@torch.no_grad()
def free_generation_metrics(
    model: S4ProteinModel,
    tokenizer: PlastidTokenizer,
    dataset: PlastidAutocompleteDataset,
    device: torch.device,
    max_windows: int,
    prompt_length: int,
    generate_length: int,
) -> dict[str, float]:
    model.eval()
    total_matches = 0
    total_bases = 0
    max_generated_run = 0
    total_n = 0
    evaluated = 0
    forbidden = [
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.eos_token_id,
    ]
    if "N" in tokenizer.vocab:
        forbidden.append(tokenizer.vocab["N"])

    for idx in range(min(max_windows, len(dataset))):
        tokens = dataset.window_tokens(idx)
        current_prompt_length = min(prompt_length, max(1, len(tokens) - 1))
        current_generate_length = min(generate_length, len(tokens) - current_prompt_length)
        if current_generate_length <= 0:
            continue

        prompt_ids = torch.tensor([tokens[:current_prompt_length]], dtype=torch.long, device=device)
        output = model.generate(
            prompt_ids,
            max_new_tokens=current_generate_length,
            temperature=1.0,
            top_k=None,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=False,
            forbidden_token_ids=tuple(forbidden),
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            min_new_tokens=current_generate_length,
        )
        generated = output[0, current_prompt_length : current_prompt_length + current_generate_length].tolist()
        truth = tokens[current_prompt_length : current_prompt_length + current_generate_length]
        total_matches += sum(int(a == b) for a, b in zip(generated, truth))
        total_bases += len(truth)
        generated_text = tokenizer.decode(generated, stop_at_eos=False)
        max_generated_run = max(max_generated_run, longest_run(generated_text))
        total_n += generated_text.count("N")
        evaluated += 1

    accuracy = total_matches / total_bases if total_bases else 0.0
    return {
        "windows": evaluated,
        "bases": total_bases,
        "accuracy": accuracy,
        "score": 1.0 - accuracy,
        "longest_generated_run": max_generated_run,
        "n_count": total_n,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    tokenizer = PlastidTokenizer(include_n=False)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        resume_config = resume_checkpoint.get("model_config", {})
        resume_variant = resume_config.get("model_variant", "legacy")
        if resume_variant != args.model_variant:
            raise ValueError(
                f"Checkpoint uses model_variant={resume_variant}; "
                f"rerun with --model_variant {resume_variant} to resume it"
            )
        tokenizer = tokenizer_from_checkpoint(resume_checkpoint)

    print("Loading plastid FASTA records...", flush=True)
    records = load_plastid_records(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        max_records=args.max_records,
        exclude_accessions=set(args.exclude_accession or []),
    )
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    train_max_windows = None
    val_max_windows = None
    if args.max_windows is not None:
        val_max_windows = max(1, int(round(args.max_windows * args.val_fraction)))
        train_max_windows = max(1, args.max_windows - val_max_windows)

    print("Building plastid FASTA windows...", flush=True)
    train_dataset = PlastidAutocompleteDataset(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        l_max=args.l_max,
        stride=args.stride,
        max_records=None,
        max_windows=train_max_windows,
        windows_per_record=args.windows_per_record,
        prefix_min_fraction=args.prefix_min_fraction,
        prefix_max_fraction=args.prefix_max_fraction,
        seed=args.seed,
        records=train_records,
    )
    val_dataset = PlastidAutocompleteDataset(
        fasta_file=args.fasta_file,
        tokenizer=tokenizer,
        l_max=args.l_max,
        stride=args.stride,
        max_records=None,
        max_windows=val_max_windows,
        windows_per_record=args.windows_per_record,
        prefix_min_fraction=args.prefix_min_fraction,
        prefix_max_fraction=args.prefix_max_fraction,
        seed=args.seed + 1,
        records=val_records,
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
        model_variant=args.model_variant,
        l_max=args.l_max,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.l_max,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.ssm_lr)
    start_epoch = 0
    best_val = float("inf")
    best_free = float("inf")

    if args.resume:
        ckpt = resume_checkpoint
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val_loss", best_val)
        best_free = ckpt.get("best_free_score", best_free)

    print("Plastid S4 run")
    print(f"  Device: {device}")
    print(f"  FASTA: {args.fasta_file}")
    print(f"  Records after exclusion/filtering: {len(records)}")
    print(f"  Excluded accessions: {', '.join(args.exclude_accession or [])}")
    print(f"  Train/val records: {len(train_records)}/{len(val_records)}")
    print(f"  Train/val windows: {len(train_dataset)}/{len(val_dataset)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model: d_model={args.d_model}, d_state={args.d_state}, n_layers={args.n_layers}")
    print(f"  Model variant: {args.model_variant}")
    if args.model_variant == "s4d_v2":
        print(f"  SSM learning rate: {args.lr if args.ssm_lr is None else args.ssm_lr}")
    print(f"  Rollout length: {args.rollout_length}")
    print(f"  Rollout loss weight: {args.rollout_loss_weight}")
    print(f"  Scheduled sampling max: {args.scheduled_sampling_max}")
    print(f"  Free eval windows: {args.free_eval_windows}")
    if args.resume:
        print(f"  Resume: {args.resume} starting at epoch {start_epoch}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_offset = epoch - start_epoch
        scheduled_sampling_probability = scheduled_sampling_for_epoch(
            epoch_offset,
            args.epochs,
            args.scheduled_sampling_max,
        )
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            args.rollout_length,
            args.rollout_loss_weight,
            scheduled_sampling_probability,
        )
        train_metrics = evaluate(model, train_loader, device) if args.eval_train else None
        metrics = evaluate(model, val_loader, device)
        free_metrics = free_generation_metrics(
            model,
            tokenizer,
            val_dataset,
            device,
            args.free_eval_windows,
            args.free_eval_prompt_length,
            args.free_eval_generate_length,
        )
        is_best = metrics["loss"] < best_val
        best_val = min(best_val, metrics["loss"])
        save = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "objective": "autocomplete",
            "data_type": "plastid_dna",
            "best_val_loss": best_val,
            "best_free_score": min(best_free, free_metrics["score"]),
            "free_generation_metrics": free_metrics,
            "scheduled_sampling_probability": scheduled_sampling_probability,
            "tokenizer_vocab": tokenizer.vocab,
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": model.d_model,
                "d_state": model.d_state,
                "n_layers": model.n_layers,
                "kernel_type": model.kernel_type,
                "model_variant": model.model_variant,
                "eos_token_id": tokenizer.eos_token_id,
                "l_max": args.l_max,
                "rollout_length": args.rollout_length,
                "rollout_loss_weight": args.rollout_loss_weight,
                "scheduled_sampling_max": args.scheduled_sampling_max,
            },
        }
        torch.save(save, os.path.join(args.output_dir, "last.pt"))
        if is_best or not os.path.exists(os.path.join(args.output_dir, "best_loss.pt")):
            torch.save(save, os.path.join(args.output_dir, "best_loss.pt"))
        if free_metrics["score"] < best_free or not os.path.exists(os.path.join(args.output_dir, "best_free.pt")):
            best_free = free_metrics["score"]
            save["best_free_score"] = best_free
            torch.save(save, os.path.join(args.output_dir, "best_free.pt"))
        line = (
            f"Epoch {epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_ppl={metrics['perplexity']:.3f} "
            f"top1={metrics['top1']:.4f} "
            f"top3={metrics['top3']:.4f} "
            f"free_acc={free_metrics['accuracy']:.4f} "
            f"free_longest_run={free_metrics['longest_generated_run']} "
            f"sched_sample={scheduled_sampling_probability:.3f}"
        )
        if train_metrics is not None:
            line += (
                f" train_top1={train_metrics['top1']:.4f}"
                f" train_top3={train_metrics['top3']:.4f}"
            )
        print(line)

    prompt = train_dataset.first_sequence[: args.prompt_length]
    generated = generate_sample(model, tokenizer, prompt, args, device)
    exact_acc = continuation_accuracy(prompt, generated, train_dataset.first_sequence)
    print()
    print("Prompt header:", train_dataset.first_header)
    print("Prompt:", prompt)
    print("Generated:", generated)
    if exact_acc is not None:
        print(f"Exact continuation accuracy: {exact_acc:.4f}")
    print("Saved:", os.path.join(args.output_dir, "best_loss.pt"))


if __name__ == "__main__":
    main()
