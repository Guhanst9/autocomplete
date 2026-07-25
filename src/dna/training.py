import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dna.checkpoint import get_device, tokenizer_from_checkpoint
from src.dna.data import DnaTokenizer, DnaWindowDataset, load_dna_records, split_records
from src.models.s4_model import S4SequenceModel


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    d_model: int
    d_state: int
    n_layers: int
    l_max: int
    max_records: Optional[int]
    max_windows: Optional[int]
    windows_per_record: Optional[int]
    epochs: int
    batch_size: int
    lr: float
    dropout: float
    seed: int
    recovery_enabled: bool
    val_fraction: float = 0.1
    stride: Optional[int] = None
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    prefix_min_fraction: float = 0.25
    prefix_max_fraction: float = 0.70
    homopolymer_loss_weight: float = 0.02
    homopolymer_min_run: int = 8
    recovery_loss_weight: float = 0.25
    recovery_start_probability: float = 0.02
    recovery_max_probability: float = 0.10
    recovery_warmup_epochs: int = 2
    free_eval_windows: int = 8
    free_eval_prompt_length: int = 512
    free_eval_generate_length: int = 64


PRESETS = {
    "quick-control": TrainingPreset(
        name="quick-control",
        d_model=196,
        d_state=64,
        n_layers=10,
        l_max=1024,
        max_records=7500,
        max_windows=None,
        windows_per_record=1,
        epochs=2,
        batch_size=2,
        lr=3e-4,
        dropout=0.1,
        seed=13,
        recovery_enabled=False,
    ),
    "quick": TrainingPreset(
        name="quick",
        d_model=196,
        d_state=64,
        n_layers=10,
        l_max=1024,
        max_records=7500,
        max_windows=None,
        windows_per_record=1,
        epochs=2,
        batch_size=2,
        lr=3e-4,
        dropout=0.1,
        seed=13,
        recovery_enabled=True,
    ),
    "full": TrainingPreset(
        name="full",
        d_model=400,
        d_state=64,
        n_layers=10,
        l_max=1024,
        max_records=None,
        max_windows=30000,
        windows_per_record=2,
        epochs=6,
        batch_size=2,
        lr=3e-4,
        dropout=0.1,
        seed=13,
        recovery_enabled=True,
    ),
}


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_sequence_model(tokenizer: DnaTokenizer, preset: TrainingPreset) -> S4SequenceModel:
    return S4SequenceModel(
        vocab_size=tokenizer.vocab_size,
        d_model=preset.d_model,
        d_state=preset.d_state,
        n_layers=preset.n_layers,
        dropout=preset.dropout,
        kernel_type="diag",
        bidirectional=False,
        model_variant="s4d_v2",
        l_max=preset.l_max,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=preset.l_max,
    )


def build_optimizer(
    model: S4SequenceModel,
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


def homopolymer_end_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    base_token_ids: set[int],
    min_run: int,
) -> tuple[torch.Tensor, int]:
    if min_run < 2:
        raise ValueError("homopolymer_min_run must be at least 2")

    inputs = input_ids.detach().cpu()
    targets = target_ids.detach().cpu()
    scored = loss_mask.detach().cpu().bool()
    selected = torch.zeros_like(scored)

    for batch_idx in range(inputs.size(0)):
        previous = None
        run_length = 0
        for pos in range(inputs.size(1)):
            token = int(inputs[batch_idx, pos])
            if token not in base_token_ids:
                previous = None
                run_length = 0
                continue
            if token == previous:
                run_length += 1
            else:
                previous = token
                run_length = 1
            if (
                run_length >= min_run
                and bool(scored[batch_idx, pos])
                and int(targets[batch_idx, pos]) != token
            ):
                selected[batch_idx, pos] = True

    position_count = int(selected.sum().item())
    if position_count == 0:
        return logits.sum() * 0.0, 0

    selected = selected.to(logits.device)
    repeat_ids = input_ids[selected].unsqueeze(-1)
    repeat_probabilities = logits[selected].softmax(dim=-1).gather(-1, repeat_ids).squeeze(-1)
    penalty = -torch.log((1.0 - repeat_probabilities).clamp_min(1e-6)).mean()
    return penalty, position_count


def recovery_probability_for_epoch(epoch_offset: int, preset: TrainingPreset) -> float:
    if not preset.recovery_enabled:
        return 0.0
    if preset.recovery_warmup_epochs <= 0:
        return preset.recovery_max_probability
    progress = min(max(epoch_offset, 0), preset.recovery_warmup_epochs) / preset.recovery_warmup_epochs
    return preset.recovery_start_probability + (
        preset.recovery_max_probability - preset.recovery_start_probability
    ) * progress


def build_recovery_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    logits: torch.Tensor,
    tokenizer: DnaTokenizer,
    corruption_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if corruption_probability <= 0:
        empty_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        return input_ids, empty_mask, 0, 0

    base_token_ids = torch.tensor(
        [tokenizer.vocab[base] for base in "ACGT"],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    is_base = (input_ids.unsqueeze(-1) == base_token_ids).any(dim=-1)

    suffix_token_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    suffix_token_mask[:, 1:] = loss_mask[:, :-1].bool()
    eligible = is_base & attention_mask.bool() & suffix_token_mask

    random_mask = torch.rand(input_ids.shape, device=input_ids.device) < corruption_probability
    selected = eligible & random_mask
    if not selected.any().item():
        empty_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        return input_ids, empty_mask, 0, int(eligible.sum().item())

    base_logits = logits.detach().clone()
    allowed = torch.zeros(base_logits.size(-1), dtype=torch.bool, device=base_logits.device)
    allowed[base_token_ids.long()] = True
    base_logits[..., ~allowed] = -1e10
    predicted_at_position = base_logits.argmax(dim=-1)

    shifted_prediction = input_ids.clone()
    shifted_prediction[:, 1:] = predicted_at_position[:, :-1]
    changed = selected & (shifted_prediction != input_ids)

    corrupted_ids = input_ids.clone()
    corrupted_ids[changed] = shifted_prediction[changed]

    history_has_corruption = changed.long().cumsum(dim=1).bool()
    recovery_mask = loss_mask.bool() & history_has_corruption
    return corrupted_ids, recovery_mask, int(changed.sum().item()), int(eligible.sum().item())


def masked_autocomplete_loss(
    model: S4SequenceModel,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if not loss_mask.bool().any().item():
        return logits.sum() * 0.0
    return model.compute_loss(
        input_ids,
        target_ids,
        attention_mask,
        loss_mask=loss_mask,
        objective="autocomplete",
        logits=logits,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    preset: TrainingPreset,
    tokenizer: DnaTokenizer,
    epoch_offset: int,
):
    model.train()
    total_loss = 0.0
    total_clean_loss = 0.0
    total_recovery_loss = 0.0
    total_homopolymer_loss = 0.0
    homopolymer_batches = 0
    homopolymer_positions = 0
    recovery_batches = 0
    corrupted_tokens = 0
    eligible_recovery_tokens = 0
    base_token_ids = {tokenizer.vocab[base] for base in "ACGT"}
    recovery_probability = recovery_probability_for_epoch(epoch_offset, preset)
    for batch in tqdm(loader, desc="Train"):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask=attention_mask)
        clean_loss = masked_autocomplete_loss(
            model,
            logits,
            input_ids,
            target_ids,
            attention_mask,
            loss_mask,
        )
        loss = clean_loss
        recovery_loss = logits.sum() * 0.0
        if preset.recovery_enabled:
            corrupted_ids, recovery_mask, changed_count, eligible_count = build_recovery_batch(
                input_ids,
                attention_mask,
                loss_mask,
                logits,
                tokenizer,
                recovery_probability,
            )
            corrupted_tokens += changed_count
            eligible_recovery_tokens += eligible_count
            if recovery_mask.any().item():
                recovery_logits = model(corrupted_ids, attention_mask=attention_mask)
                recovery_loss = masked_autocomplete_loss(
                    model,
                    recovery_logits,
                    corrupted_ids,
                    target_ids,
                    attention_mask,
                    recovery_mask.long(),
                )
                loss = loss + preset.recovery_loss_weight * recovery_loss
                recovery_batches += 1
        if preset.homopolymer_loss_weight > 0:
            homopolymer_loss, position_count = homopolymer_end_loss(
                logits,
                input_ids,
                target_ids,
                loss_mask,
                base_token_ids,
                preset.homopolymer_min_run,
            )
            loss = loss + preset.homopolymer_loss_weight * homopolymer_loss
            if position_count > 0:
                total_homopolymer_loss += homopolymer_loss.item()
                homopolymer_batches += 1
                homopolymer_positions += position_count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_clean_loss += clean_loss.item()
        total_recovery_loss += recovery_loss.item()
    return {
        "loss": total_loss / max(1, len(loader)),
        "clean_loss": total_clean_loss / max(1, len(loader)),
        "recovery_loss": total_recovery_loss / max(1, len(loader)),
        "recovery_probability": recovery_probability,
        "recovery_batches": recovery_batches,
        "corrupted_token_fraction": corrupted_tokens / max(1, eligible_recovery_tokens),
        "homopolymer_loss": total_homopolymer_loss / max(1, homopolymer_batches),
        "homopolymer_positions": homopolymer_positions,
    }


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
            logits=logits,
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
    model: S4SequenceModel,
    tokenizer: DnaTokenizer,
    dataset: DnaWindowDataset,
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
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=False,
            forbidden_token_ids=tuple(forbidden),
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


def assert_resume_matches_preset(checkpoint: dict, preset: TrainingPreset) -> None:
    config = checkpoint.get("model_config", {})
    expected = {
        "d_model": preset.d_model,
        "d_state": preset.d_state,
        "n_layers": preset.n_layers,
        "kernel_type": "diag",
        "model_variant": "s4d_v2",
        "l_max": preset.l_max,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}=checkpoint:{old} preset:{new}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"Resume checkpoint does not match preset {preset.name}: {details}")


def build_datasets(
    fasta_file: str,
    tokenizer: DnaTokenizer,
    preset: TrainingPreset,
    holdout_accession: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], DnaWindowDataset, DnaWindowDataset]:
    records = load_dna_records(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        max_records=preset.max_records,
        exclude_accessions={holdout_accession},
    )
    train_records, val_records = split_records(records, preset.val_fraction, preset.seed)
    train_max_windows = None
    val_max_windows = None
    if preset.max_windows is not None:
        val_max_windows = max(1, int(round(preset.max_windows * preset.val_fraction)))
        train_max_windows = max(1, preset.max_windows - val_max_windows)

    train_dataset = DnaWindowDataset(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        l_max=preset.l_max,
        stride=preset.stride,
        max_windows=train_max_windows,
        windows_per_record=preset.windows_per_record,
        prefix_min_fraction=preset.prefix_min_fraction,
        prefix_max_fraction=preset.prefix_max_fraction,
        seed=preset.seed,
        records=train_records,
    )
    val_dataset = DnaWindowDataset(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        l_max=preset.l_max,
        stride=preset.stride,
        max_windows=val_max_windows,
        windows_per_record=preset.windows_per_record,
        prefix_min_fraction=preset.prefix_min_fraction,
        prefix_max_fraction=preset.prefix_max_fraction,
        seed=preset.seed + 1,
        records=val_records,
    )
    return train_records, val_records, train_dataset, val_dataset


def run_training(
    preset_name: str,
    fasta_file: str,
    output_dir: str,
    resume: Optional[str],
    holdout_accession: str,
) -> None:
    preset = PRESETS[preset_name]
    random.seed(preset.seed)
    torch.manual_seed(preset.seed)
    device = get_device()
    tokenizer = DnaTokenizer(include_n=False)
    resume_checkpoint = None
    if resume:
        resume_checkpoint = torch.load(resume, map_location=device)
        assert_resume_matches_preset(resume_checkpoint, preset)
        tokenizer = tokenizer_from_checkpoint(resume_checkpoint)
        if "N" in tokenizer.vocab:
            raise ValueError("New DNA training checkpoints must use an A/C/G/T tokenizer without N")

    print("Loading DNA FASTA records...", flush=True)
    train_records, val_records, train_dataset, val_dataset = build_datasets(
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        preset=preset,
        holdout_accession=holdout_accession,
    )
    train_loader = DataLoader(train_dataset, batch_size=preset.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=preset.batch_size, shuffle=False, num_workers=0)

    model = build_sequence_model(tokenizer, preset).to(device)
    optimizer = build_optimizer(model, preset.lr, preset.weight_decay)
    start_epoch = 0
    best_val = float("inf")
    best_free = float("inf")

    if resume:
        ckpt = resume_checkpoint
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val_loss", best_val)
        best_free = ckpt.get("best_free_score", best_free)

    print("DNA S4D run")
    print(f"  Device: {device}")
    print(f"  Preset: {preset.name}")
    print(f"  FASTA: {fasta_file}")
    print(f"  Holdout accession: {holdout_accession}")
    print(f"  Train/val records: {len(train_records)}/{len(val_records)}")
    print(f"  Train/val windows: {len(train_dataset)}/{len(val_dataset)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model: d_model={preset.d_model}, d_state={preset.d_state}, n_layers={preset.n_layers}")
    print(f"  Model variant: s4d_v2")
    print(f"  Parameters: {parameter_count(model):,}")
    print(f"  Recovery enabled: {'yes' if preset.recovery_enabled else 'no'}")
    print(f"  Homopolymer loss weight: {preset.homopolymer_loss_weight}")
    print(f"  Homopolymer minimum run: {preset.homopolymer_min_run}")
    print(f"  Free eval windows: {preset.free_eval_windows}")
    if resume:
        print(f"  Resume: {resume} starting at epoch {start_epoch}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + preset.epochs):
        train_epoch_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            preset,
            tokenizer,
            epoch - start_epoch,
        )
        train_loss = train_epoch_metrics["loss"]
        metrics = evaluate(model, val_loader, device)
        free_metrics = free_generation_metrics(
            model,
            tokenizer,
            val_dataset,
            device,
            preset.free_eval_windows,
            preset.free_eval_prompt_length,
            preset.free_eval_generate_length,
        )
        is_best = metrics["loss"] < best_val
        best_val = min(best_val, metrics["loss"])
        save = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "objective": "autocomplete",
            "data_type": "plastid_dna",
            "preset": preset.name,
            "preset_config": asdict(preset),
            "holdout_accession": holdout_accession,
            "best_val_loss": best_val,
            "best_free_score": min(best_free, free_metrics["score"]),
            "free_generation_metrics": free_metrics,
            "recovery_settings": {
                "enabled": preset.recovery_enabled,
                "loss_weight": preset.recovery_loss_weight,
                "start_probability": preset.recovery_start_probability,
                "max_probability": preset.recovery_max_probability,
                "warmup_epochs": preset.recovery_warmup_epochs,
                "current_probability": train_epoch_metrics["recovery_probability"],
            },
            "tokenizer_vocab": tokenizer.vocab,
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": model.d_model,
                "d_state": model.d_state,
                "n_layers": model.n_layers,
                "kernel_type": model.kernel_type,
                "model_variant": model.model_variant,
                "eos_token_id": tokenizer.eos_token_id,
                "l_max": preset.l_max,
                "dropout": preset.dropout,
                "homopolymer_loss_weight": preset.homopolymer_loss_weight,
                "homopolymer_min_run": preset.homopolymer_min_run,
                "recovery_enabled": preset.recovery_enabled,
                "recovery_loss_weight": preset.recovery_loss_weight,
                "recovery_start_probability": preset.recovery_start_probability,
                "recovery_max_probability": preset.recovery_max_probability,
                "recovery_warmup_epochs": preset.recovery_warmup_epochs,
            },
        }
        torch.save(save, os.path.join(output_dir, "last.pt"))
        if is_best or not os.path.exists(os.path.join(output_dir, "best_loss.pt")):
            torch.save(save, os.path.join(output_dir, "best_loss.pt"))
        if free_metrics["score"] < best_free or not os.path.exists(os.path.join(output_dir, "best_free.pt")):
            best_free = free_metrics["score"]
            save["best_free_score"] = best_free
            torch.save(save, os.path.join(output_dir, "best_free.pt"))
        print(
            f"Epoch {epoch} "
            f"train_loss={train_loss:.4f} "
            f"clean_loss={train_epoch_metrics['clean_loss']:.4f} "
            f"recovery_loss={train_epoch_metrics['recovery_loss']:.4f} "
            f"recovery_p={train_epoch_metrics['recovery_probability']:.3f} "
            f"corrupt_frac={train_epoch_metrics['corrupted_token_fraction']:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_ppl={metrics['perplexity']:.3f} "
            f"top1={metrics['top1']:.4f} "
            f"top3={metrics['top3']:.4f} "
            f"free_acc={free_metrics['accuracy']:.4f} "
            f"free_longest_run={free_metrics['longest_generated_run']} "
            f"hpoly_loss={train_epoch_metrics['homopolymer_loss']:.4f} "
            f"hpoly_positions={train_epoch_metrics['homopolymer_positions']}"
        )

    print()
    print("Saved:", os.path.join(output_dir, "best_loss.pt"))
