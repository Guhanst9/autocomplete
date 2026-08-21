import argparse
import gc
import json
import math
import random
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dna.data import DnaTokenizer
from src.dna.prediction import TripletCodec
from src.dna.training import (
    PRESETS,
    build_datasets,
    build_optimizer,
    build_sequence_model,
    parameter_count,
    train_one_epoch,
)


MIB = 1024**2


class NvidiaSampler:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
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
            self.stop_event.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)


def recovery_presets(batch_size: int):
    base = replace(PRESETS["size-current"], batch_size=batch_size)
    return {
        "independent": base,
        "recursive-24": replace(
            base,
            recovery_corruption_mode="recursive-block",
            recursive_recovery_block_length=24,
        ),
        "recursive-48": replace(
            base,
            recovery_corruption_mode="recursive-block",
            recursive_recovery_block_length=48,
        ),
    }


def configure_model(model, codec):
    model.prediction_unit = "triplet"
    model.bases_per_prediction = 3
    model.output_tokens = codec.triplets


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_mode(name, preset, dataset, tokenizer, codec, device, steps, warmup_steps):
    random.seed(preset.seed)
    torch.manual_seed(preset.seed)
    torch.cuda.empty_cache()
    model = build_sequence_model(tokenizer, preset, prediction_unit="triplet").to(device)
    configure_model(model, codec)
    optimizer = build_optimizer(model, preset.lr, preset.weight_decay)
    generator = torch.Generator().manual_seed(preset.seed)
    loader = DataLoader(
        dataset,
        batch_size=preset.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    iterator = iter(loader)
    warmup = list(islice(iterator, warmup_steps))
    measured = list(islice(iterator, steps))
    train_one_epoch(
        model,
        warmup,
        optimizer,
        device,
        preset,
        tokenizer,
        2,
        "triplet",
        codec,
    )

    synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with NvidiaSampler() as sampler:
        started = time.perf_counter()
        metrics = train_one_epoch(
            model,
            measured,
            optimizer,
            device,
            preset,
            tokenizer,
            2,
            "triplet",
            codec,
        )
        synchronize()
        elapsed = time.perf_counter() - started

    iterations_per_second = len(measured) / elapsed
    estimated_epoch_seconds = math.ceil(len(dataset) / preset.batch_size) / iterations_per_second
    result = {
        "mode": name,
        "steps": len(measured),
        "seconds": elapsed,
        "iterations_per_second": iterations_per_second,
        "examples_per_second": len(measured) * preset.batch_size / elapsed,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "peak_nvidia_smi_memory_mib": max((row[1] for row in sampler.samples), default=0.0),
        "average_gpu_utilization_percent": statistics.fmean(row[0] for row in sampler.samples)
        if sampler.samples
        else None,
        "recursive_batches": metrics["recursive_recovery_batches"],
        "loss": metrics["loss"],
        "preset": asdict(preset),
    }
    del optimizer, model, measured, warmup
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark detached recursive recovery on CUDA.")
    parser.add_argument(
        "--fasta-file",
        default="data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz",
    )
    parser.add_argument("--holdout-accession", default="NC_053550.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hourly-price", type=float, required=True)
    args = parser.parse_args()
    if args.steps < 100 or args.warmup_steps <= 0 or args.batch_size <= 0:
        parser.error(
            "use at least 100 measured steps, positive warmup steps, "
            "and a positive batch size"
        )
    if args.hourly_price <= 0:
        parser.error("hourly price must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("recursive recovery benchmark requires CUDA")

    device = torch.device("cuda")
    tokenizer = DnaTokenizer(include_n=False)
    codec = TripletCodec()
    base_preset = replace(PRESETS["size-current"], batch_size=args.batch_size)
    _, _, train_dataset, _ = build_datasets(
        args.fasta_file,
        tokenizer,
        base_preset,
        args.holdout_accession,
        "triplet",
        codec,
    )
    results = []
    for name, preset in recovery_presets(args.batch_size).items():
        print(f"Benchmarking {name}", flush=True)
        results.append(
            run_mode(
                name,
                preset,
                train_dataset,
                tokenizer,
                codec,
                device,
                args.steps,
                args.warmup_steps,
            )
        )

    for result in results:
        result["estimated_epoch_cost"] = (
            result["estimated_epoch_seconds"] / 3600 * args.hourly_price
        )
    output = {
        "device": torch.cuda.get_device_name(0),
        "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / MIB,
        "hourly_price": args.hourly_price,
        "parameters": parameter_count(
            build_sequence_model(tokenizer, base_preset, prediction_unit="triplet")
        ),
        "train_windows": len(train_dataset),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
