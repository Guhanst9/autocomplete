import argparse
import os
import time

import torch

from src.dna.data import DnaTokenizer
from src.dna.training import (
    TRANSFORMER_PRESETS,
    build_optimizer,
    build_recovery_batch,
    build_sequence_model,
    homopolymer_end_loss,
    masked_autocomplete_loss,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Transformer training on MPS.")
    parser.add_argument("--preset", choices=sorted(TRANSFORMER_PRESETS), default="full")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("disable PYTORCH_ENABLE_MPS_FALLBACK before benchmarking")
    if args.warmup_steps < 0 or args.steps <= 0:
        raise ValueError("warmup-steps must be nonnegative and steps must be positive")

    preset = TRANSFORMER_PRESETS[args.preset]
    tokenizer = DnaTokenizer()
    device = torch.device("mps")
    model = build_sequence_model(tokenizer, preset, model_type="transformer").to(device)
    optimizer = build_optimizer(model, preset.lr, preset.weight_decay)
    base_ids = torch.tensor(
        [tokenizer.vocab[base] for base in "ACGT"],
        dtype=torch.long,
        device=device,
    )
    choices = torch.randint(
        0,
        len(base_ids),
        (preset.batch_size, preset.l_max),
        device=device,
    )
    input_ids = base_ids[choices]
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    loss_mask[:, : preset.l_max // 4] = 0
    loss_mask[:, -1] = 0
    base_token_ids = {tokenizer.vocab[base] for base in "ACGT"}

    def step() -> float:
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
        corrupted_ids, recovery_mask, _, _ = build_recovery_batch(
            input_ids,
            attention_mask,
            loss_mask,
            logits,
            tokenizer,
            preset.recovery_max_probability,
        )
        recovery_logits = model(corrupted_ids, attention_mask=attention_mask)
        recovery_loss = masked_autocomplete_loss(
            model,
            recovery_logits,
            corrupted_ids,
            target_ids,
            attention_mask,
            recovery_mask.long(),
        )
        homopolymer_loss, _ = homopolymer_end_loss(
            logits,
            input_ids,
            target_ids,
            loss_mask,
            base_token_ids,
            preset.homopolymer_min_run,
        )
        loss = (
            clean_loss
            + preset.recovery_loss_weight * recovery_loss
            + preset.homopolymer_loss_weight * homopolymer_loss
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("benchmark produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
        optimizer.step()
        return loss.item()

    for _ in range(args.warmup_steps):
        step()
    torch.mps.synchronize()
    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.steps):
        final_loss = step()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    seconds_per_step = elapsed / args.steps
    full_batches = 13_500 if args.preset == "full" else None

    print("Transformer MPS benchmark")
    print(f"  Preset: {args.preset}")
    print(f"  Parameters: {parameter_count(model):,}")
    print(f"  Batch/context: {preset.batch_size}/{preset.l_max}")
    print(f"  Seconds per training step: {seconds_per_step:.3f}")
    if full_batches is not None:
        print(f"  Estimated full epoch: {seconds_per_step * full_batches / 3600:.2f} hours")
    print(f"  Current MPS allocation: {torch.mps.current_allocated_memory() / 2**30:.2f} GiB")
    print(f"  Driver MPS allocation: {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB")
    print(f"  Final synthetic loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
