import argparse
import gc
import json
import math
import random
import statistics
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dna.data import DnaTokenizer
from src.dna.prediction import TripletCodec
from src.dna.training import (
    PRESETS,
    build_datasets,
    build_optimizer,
    build_recovery_batch,
    build_sequence_model,
    free_generation_metrics,
    homopolymer_end_loss,
    masked_autocomplete_loss,
    parameter_count,
    record_fingerprint,
    recovery_probability_for_epoch,
    unpack_batch,
    window_fingerprint,
)


SIZE_PRESETS = ("size-small", "size-current", "size-large")
DEFAULT_BATCH_SIZES = (2, 4, 8)
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark controlled triplet S4D model sizes.")
    parser.add_argument(
        "--fasta-file",
        default="data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz",
    )
    parser.add_argument("--holdout-accession", default="NC_053550.1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    args = parser.parse_args()
    if args.steps < 50:
        parser.error("--steps must be at least 50")
    if args.warmup_steps < 1 or args.validation_steps < 1:
        parser.error("warmup and validation steps must be positive")
    if any(size <= 0 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    return args


def configure_model(model, codec: TripletCodec) -> None:
    model.prediction_unit = "triplet"
    model.bases_per_prediction = 3
    model.output_tokens = codec.triplets


def training_step(
    model,
    optimizer,
    batch,
    device,
    preset,
    tokenizer,
    codec,
    recovery_probability,
) -> dict[str, float]:
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
    corrupted_ids, recovery_mask, _, _ = build_recovery_batch(
        input_ids,
        attention_mask,
        loss_mask,
        logits,
        tokenizer,
        recovery_probability,
        prediction_unit="triplet",
        triplet_codec=codec,
        corruption_mode=preset.recovery_corruption_mode,
        block_min_length=preset.recovery_block_min_length,
        block_max_length=preset.recovery_block_max_length,
    )
    recovery_loss = logits.sum() * 0.0
    loss = clean_loss
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

    base_token_ids = {tokenizer.vocab[base] for base in "ACGT"}
    homopolymer_loss, homopolymer_positions = homopolymer_end_loss(
        logits,
        input_ids,
        target_ids,
        loss_mask,
        base_token_ids,
        preset.homopolymer_min_run,
        prediction_unit="triplet",
        triplet_codec=codec,
        tokenizer=tokenizer,
    )
    loss = loss + preset.homopolymer_loss_weight * homopolymer_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
    optimizer.step()
    return {
        "total_loss": float(loss.detach()),
        "clean_loss": float(clean_loss.detach()),
        "recovery_loss": float(recovery_loss.detach()),
        "homopolymer_loss": float(homopolymer_loss.detach()),
        "homopolymer_positions": homopolymer_positions,
    }


class NvidiaSampler:
    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                utilization, memory = result.stdout.strip().split(",")
                self.samples.append((float(utilization), float(memory)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join(timeout=2)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def benchmark_validation(model, dataset, batch_size: int, steps: int, device) -> dict:
    model.eval()
    random.seed(13)
    loader = make_loader(dataset, batch_size, shuffle=False, seed=13)
    iterator = iter(loader)
    torch.cuda.synchronize()
    started = time.perf_counter()
    completed = 0
    for _ in range(min(steps, len(loader))):
        input_ids, target_ids, attention_mask, loss_mask = unpack_batch(next(iterator), device)
        logits = model(input_ids, attention_mask=attention_mask)
        model.compute_loss(
            input_ids,
            target_ids,
            attention_mask,
            loss_mask=loss_mask,
            objective="autocomplete",
            logits=logits,
        )
        completed += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": completed,
        "seconds": elapsed,
        "iterations_per_second": completed / elapsed,
        "examples_per_second": completed * batch_size / elapsed,
    }


def benchmark_combination(
    preset_name: str,
    batch_size: int,
    train_dataset,
    val_dataset,
    tokenizer,
    codec,
    device,
    steps: int,
    warmup_steps: int,
    validation_steps: int,
) -> dict:
    preset = PRESETS[preset_name]
    result = {
        "preset": preset_name,
        "batch_size": batch_size,
        "learning_rate": preset.lr,
        "measured_steps": steps,
        "warmup_steps": warmup_steps,
        "status": "pending",
    }
    model = optimizer = batch = None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(preset.seed)
        model = build_sequence_model(tokenizer, preset, prediction_unit="triplet").to(device)
        configure_model(model, codec)
        optimizer = build_optimizer(model, preset.lr, preset.weight_decay)
        model.train()
        random.seed(preset.seed)
        torch.manual_seed(10_000 + batch_size)
        loader = make_loader(train_dataset, batch_size, shuffle=True, seed=preset.seed)
        iterator = iter(loader)
        recovery_probability = recovery_probability_for_epoch(2, preset)

        for _ in range(warmup_steps):
            batch = next(iterator)
            training_step(
                model,
                optimizer,
                batch,
                device,
                preset,
                tokenizer,
                codec,
                recovery_probability,
            )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        measured = []
        with NvidiaSampler() as sampler:
            started = time.perf_counter()
            for _ in range(steps):
                batch = next(iterator)
                measured.append(
                    training_step(
                        model,
                        optimizer,
                        batch,
                        device,
                        preset,
                        tokenizer,
                        codec,
                        recovery_probability,
                    )
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

        peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
        peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
        utilization = [sample[0] for sample in sampler.samples]
        sampled_memory = [sample[1] for sample in sampler.samples]
        train_iterations_per_second = steps / elapsed
        validation = benchmark_validation(
            model,
            val_dataset,
            batch_size,
            validation_steps,
            device,
        )
        result.update(
            {
                "status": "success",
                "seconds": elapsed,
                "iterations_per_second": train_iterations_per_second,
                "examples_per_second": steps * batch_size / elapsed,
                "peak_allocated_memory_mib": peak_allocated,
                "peak_reserved_memory_mib": peak_reserved,
                "peak_nvidia_smi_memory_mib": max(sampled_memory, default=peak_reserved),
                "average_gpu_utilization_percent": statistics.fmean(utilization) if utilization else None,
                "gpu_utilization_samples": len(utilization),
                "recovery_probability": recovery_probability,
                "mean_losses": {
                    key: statistics.fmean(row[key] for row in measured)
                    for key in ("total_loss", "clean_loss", "recovery_loss", "homopolymer_loss")
                },
                "homopolymer_positions": sum(row["homopolymer_positions"] for row in measured),
                "estimated_training_seconds_per_epoch": math.ceil(len(train_dataset) / batch_size)
                / train_iterations_per_second,
                "validation_probe": validation,
                "estimated_validation_seconds_per_epoch": math.ceil(len(val_dataset) / batch_size)
                / validation["iterations_per_second"],
            }
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            result.update({"status": "oom", "error": str(error)})
        else:
            raise
    finally:
        del batch, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def select_common_batch(results: list[dict], total_memory_mib: float) -> dict:
    candidates = []
    batch_sizes = sorted({row["batch_size"] for row in results})
    for batch_size in batch_sizes:
        rows = [row for row in results if row["batch_size"] == batch_size]
        if len(rows) != len(SIZE_PRESETS) or any(row["status"] != "success" for row in rows):
            continue
        peak_memory = max(
            max(row["peak_reserved_memory_mib"], row["peak_nvidia_smi_memory_mib"])
            for row in rows
        )
        headroom = total_memory_mib - peak_memory
        candidates.append(
            {
                "batch_size": batch_size,
                "mean_examples_per_second": statistics.fmean(
                    row["examples_per_second"] for row in rows
                ),
                "minimum_headroom_mib": headroom,
                "minimum_headroom_fraction": headroom / total_memory_mib,
                "meets_20_percent_headroom": headroom >= 0.20 * total_memory_mib,
            }
        )
    safe = [candidate for candidate in candidates if candidate["meets_20_percent_headroom"]]
    eligible = safe or candidates
    if not eligible:
        raise RuntimeError("no batch size completed successfully on all three models")
    selected = max(eligible, key=lambda item: item["mean_examples_per_second"])
    return {"selected": selected, "candidates": candidates}


def benchmark_free_evaluation(preset_name, val_dataset, tokenizer, codec, device) -> dict:
    preset = PRESETS[preset_name]
    torch.manual_seed(preset.seed)
    model = build_sequence_model(tokenizer, preset, prediction_unit="triplet").to(device)
    configure_model(model, codec)
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = free_generation_metrics(
        model,
        tokenizer,
        val_dataset,
        device,
        preset.free_eval_windows,
        preset.free_eval_prompt_length,
        preset.free_eval_generate_length,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"seconds": elapsed, "windows": metrics["windows"]}


def markdown_report(report: dict) -> str:
    lines = [
        "# Triplet S4D size benchmark",
        "",
        "| Preset | Parameters | Batch | Status | Iter/s | Examples/s | Peak GPU MiB | Avg GPU util | Est. train epoch |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    counts = report["parameter_counts"]
    for row in report["benchmarks"]:
        if row["status"] == "success":
            lines.append(
                f"| {row['preset']} | {counts[row['preset']]:,} | {row['batch_size']} | success | "
                f"{row['iterations_per_second']:.3f} | {row['examples_per_second']:.3f} | "
                f"{max(row['peak_reserved_memory_mib'], row['peak_nvidia_smi_memory_mib']):.0f} | "
                f"{row['average_gpu_utilization_percent']:.1f}% | "
                f"{row['estimated_training_seconds_per_epoch'] / 3600:.3f} h |"
            )
        else:
            lines.append(
                f"| {row['preset']} | {counts[row['preset']]:,} | {row['batch_size']} | "
                f"{row['status']} | - | - | - | - | - |"
            )
    selected = report["recommendation"]["selected"]
    lines.extend(
        [
            "",
            f"Recommended common batch size: **{selected['batch_size']}**.",
            "",
            f"Minimum measured headroom: {selected['minimum_headroom_mib']:.0f} MiB "
            f"({100 * selected['minimum_headroom_fraction']:.1f}%).",
            "",
            report["control_note"],
            "",
            "Estimates include measured resampling, training, validation, and free-generation overhead. "
            "They assume the maximum 20 epochs; early stopping can reduce time and cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    tokenizer = DnaTokenizer(include_n=False)
    codec = TripletCodec()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameter_counts = {}
    preset_configs = {}
    for name in SIZE_PRESETS:
        preset = PRESETS[name]
        model = build_sequence_model(tokenizer, preset, prediction_unit="triplet")
        parameter_counts[name] = parameter_count(model)
        preset_configs[name] = asdict(preset)
        del model

    invariant_snapshots = {}
    dataset_build_seconds = {}
    train_dataset = val_dataset = None
    for index, name in enumerate(SIZE_PRESETS):
        started = time.perf_counter()
        train_records, val_records, current_train, current_val = build_datasets(
            fasta_file=args.fasta_file,
            tokenizer=tokenizer,
            preset=PRESETS[name],
            holdout_accession=args.holdout_accession,
            prediction_unit="triplet",
            triplet_codec=codec,
        )
        dataset_build_seconds[name] = time.perf_counter() - started
        invariant_snapshots[name] = {
            "train_records": len(train_records),
            "validation_records": len(val_records),
            "train_windows": len(current_train),
            "validation_windows": len(current_val),
            "train_record_fingerprint": record_fingerprint(train_records),
            "validation_record_fingerprint": record_fingerprint(val_records),
            "train_window_fingerprint": window_fingerprint(current_train),
            "validation_window_fingerprint": window_fingerprint(current_val),
        }
        if index == len(SIZE_PRESETS) - 1:
            train_dataset, val_dataset = current_train, current_val
        else:
            del current_train, current_val, train_records, val_records
            gc.collect()

    first_snapshot = invariant_snapshots[SIZE_PRESETS[0]]
    if any(snapshot != first_snapshot for snapshot in invariant_snapshots.values()):
        raise RuntimeError("size presets did not produce identical data fingerprints")

    before_resample = window_fingerprint(train_dataset)
    started = time.perf_counter()
    train_dataset.resample(PRESETS[SIZE_PRESETS[0]].seed + 1)
    resample_seconds = time.perf_counter() - started
    after_resample = window_fingerprint(train_dataset)
    if before_resample == after_resample:
        raise RuntimeError("training windows did not change after resampling")
    validation_fingerprint_after_resample = window_fingerprint(val_dataset)
    if validation_fingerprint_after_resample != first_snapshot["validation_window_fingerprint"]:
        raise RuntimeError("validation windows changed while resampling training windows")

    total_memory_mib = torch.cuda.get_device_properties(device).total_memory / MIB
    benchmarks = []
    for name in SIZE_PRESETS:
        for batch_size in args.batch_sizes:
            print(f"Benchmarking {name} batch_size={batch_size}", flush=True)
            result = benchmark_combination(
                name,
                batch_size,
                train_dataset,
                val_dataset,
                tokenizer,
                codec,
                device,
                args.steps,
                args.warmup_steps,
                args.validation_steps,
            )
            benchmarks.append(result)
            print(json.dumps(result, indent=2), flush=True)

    recommendation = select_common_batch(benchmarks, total_memory_mib)
    selected_batch_size = recommendation["selected"]["batch_size"]
    free_evaluation = {
        name: benchmark_free_evaluation(name, val_dataset, tokenizer, codec, device)
        for name in SIZE_PRESETS
    }

    maximum_run_estimates = {}
    for name in SIZE_PRESETS:
        row = next(
            result
            for result in benchmarks
            if result["preset"] == name and result["batch_size"] == selected_batch_size
        )
        epoch_seconds = (
            resample_seconds
            + row["estimated_training_seconds_per_epoch"]
            + row["estimated_validation_seconds_per_epoch"]
            + free_evaluation[name]["seconds"]
        )
        maximum_run_estimates[name] = {
            "dataset_build_seconds": dataset_build_seconds[name],
            "resample_seconds_per_epoch": resample_seconds,
            "training_seconds_per_epoch": row["estimated_training_seconds_per_epoch"],
            "validation_seconds_per_epoch": row["estimated_validation_seconds_per_epoch"],
            "free_evaluation_seconds_per_epoch": free_evaluation[name]["seconds"],
            "estimated_full_epoch_seconds": epoch_seconds,
            "estimated_maximum_20_epoch_seconds": dataset_build_seconds[name] + 20 * epoch_seconds,
        }

    report = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "total_memory_mib": total_memory_mib,
        },
        "benchmark_protocol": {
            "prediction_unit": "triplet",
            "measured_training_steps": args.steps,
            "warmup_training_steps": args.warmup_steps,
            "validation_probe_steps": args.validation_steps,
            "learning_rate": 0.0003,
            "recovery_probability": 0.10,
            "losses": ["clean", "recovery", "homopolymer"],
            "includes_backward_and_optimizer_step": True,
        },
        "parameter_counts": parameter_counts,
        "preset_configs": preset_configs,
        "data_invariants": invariant_snapshots,
        "training_resample": {
            "seconds": resample_seconds,
            "before_fingerprint": before_resample,
            "after_fingerprint": after_resample,
            "validation_fingerprint_after": validation_fingerprint_after_resample,
        },
        "benchmarks": benchmarks,
        "recommendation": recommendation,
        "free_evaluation": free_evaluation,
        "maximum_run_estimates": maximum_run_estimates,
        "control_note": (
            "All three new model-size runs remain controlled because they use the same batch size. "
            "Comparison with the previous batch-size-2 model is less exact."
            if selected_batch_size > 2
            else "The selected batch size matches the previous batch-size-2 model."
        ),
    }
    (args.output_dir / "benchmark_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "benchmark_report.md").write_text(markdown_report(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
