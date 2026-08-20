#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=${1:?repository root is required}
input_root=${2:?input root is required}
output_root=${3:?output root is required}

mkdir -p "$output_root/checkpoints" "$output_root/logs" "$output_root/rosa_circular_eval"
cp "$input_root/triplet_last.pt" "$output_root/checkpoints/last.pt"
cp "$input_root/triplet_best_loss.pt" "$output_root/checkpoints/best_loss.pt"
cp "$input_root/triplet_best_loss.pt" "$output_root/checkpoints/best_loss_before_continuation.pt"

monitor_pid=""
cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$output_root/exit_status.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_root/finished_at_utc.txt"
    sudo shutdown -h +30 >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT

cd "$repo_root"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_root/started_at_utc.txt"

nvidia-smi \
    --query-gpu=timestamp,index,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    --loop=10 > "$output_root/logs/gpu_telemetry.csv" 2>&1 &
monitor_pid=$!

python - "$input_root" "$output_root" <<'PY' 2>&1 | tee "$output_root/logs/preflight.log"
import json
import sys
from pathlib import Path

import torch

from src.dna.checkpoint import load_model

input_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
paths = {
    "triplet_last": input_root / "triplet_last.pt",
    "triplet_best_loss": input_root / "triplet_best_loss.pt",
    "one_base_best_loss": input_root / "one_base_best_loss.pt",
}
report = {
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "checkpoints": {},
}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
for name, path in paths.items():
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, _, device = load_model(str(path))
    optimizer = checkpoint.get("optimizer")
    report["checkpoints"][name] = {
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "prediction_unit": getattr(model, "prediction_unit", None),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_device": str(device),
        "has_optimizer": optimizer is not None,
        "optimizer_state_entries": len(optimizer.get("state", {})) if optimizer else 0,
    }
    del model
if report["checkpoints"]["triplet_last"]["epoch"] != 5:
    raise RuntimeError("triplet last checkpoint is not epoch 5")
if not report["checkpoints"]["triplet_last"]["has_optimizer"]:
    raise RuntimeError("triplet last checkpoint has no optimizer state")
(output_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
sha256sum \
    "$input_root/triplet_last.pt" \
    "$input_root/triplet_best_loss.pt" \
    "$input_root/one_base_best_loss.pt" \
    "$input_root/refseq_plastids_all_clean_no_n.fna.gz" \
    "$input_root/one_base_rosa_sampled_t08_seed13.csv" \
    > "$output_root/input_sha256sums.txt"

training_started=$(date +%s)
/usr/bin/time -v -o "$output_root/logs/training_resource_usage.txt" \
    python run_plastid.py \
        --preset full \
        --fasta-file "$input_root/refseq_plastids_all_clean_no_n.fna.gz" \
        --output-dir "$output_root/checkpoints" \
        --resume "$output_root/checkpoints/last.pt" \
        --holdout-accession NC_053550.1 \
        --prediction-unit triplet \
        --max-additional-epochs 9 \
        --early-stopping-patience 3 \
        --early-stopping-min-delta 0.005 \
    2>&1 | tee "$output_root/logs/training_continued.log"
training_finished=$(date +%s)
printf '%s\n' "$((training_finished - training_started))" > "$output_root/training_runtime_seconds.txt"

evaluation_started=$(date +%s)
/usr/bin/time -v -o "$output_root/logs/evaluation_resource_usage.txt" \
    python run_sliding_eval.py \
        --checkpoint "$output_root/checkpoints/best_loss.pt" \
        --fasta_file "$input_root/refseq_plastids_all_clean_no_n.fna.gz" \
        --accession NC_053550.1 \
        --prompt_length 512 \
        --generate_length 512 \
        --stride 256 \
        --circular \
        --batch_size 4 \
        --decoding_mode sampled \
        --temperature 0.8 \
        --seed 13 \
        --output_dir "$output_root/rosa_circular_eval" \
    2>&1 | tee "$output_root/logs/evaluation.log"
evaluation_finished=$(date +%s)
printf '%s\n' "$((evaluation_finished - evaluation_started))" > "$output_root/evaluation_runtime_seconds.txt"

python compare_rosa_evals.py \
    --triplet-csv "$output_root/rosa_circular_eval/NC_053550.1_windows.csv" \
    --one-base-csv "$input_root/one_base_rosa_sampled_t08_seed13.csv" \
    --output "$output_root/rosa_comparison.json" \
    2>&1 | tee "$output_root/logs/comparison.log"

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""
find "$output_root" -type f ! -name sha256sums.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$output_root/sha256sums.txt"
