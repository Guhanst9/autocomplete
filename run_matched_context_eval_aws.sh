#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=${1:?repository root is required}
input_root=${2:?input root is required}
output_root=${3:?output root is required}

mkdir -p "$output_root/logs"
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

python - "$input_root/best_loss.pt" <<'PY' 2>&1 | tee "$output_root/logs/preflight.log"
import json
import sys

import torch

from src.dna.checkpoint import load_model

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model, _, device = load_model(checkpoint_path)
report = {
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "checkpoint_epoch": checkpoint.get("epoch"),
    "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
    "prediction_unit": getattr(model, "prediction_unit", None),
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "load_device": str(device),
}
if not report["cuda_available"]:
    raise RuntimeError("CUDA is not available")
if report["prediction_unit"] != "triplet":
    raise RuntimeError("checkpoint is not a triplet model")
print(json.dumps(report, indent=2))
PY

sha256sum \
    "$input_root/best_loss.pt" \
    "$input_root/baseline_512.csv" \
    "$input_root/refseq_plastids_all.fna.gz" \
    > "$output_root/input_sha256sums.txt"

python run_matched_context_eval.py \
    --checkpoint "$input_root/best_loss.pt" \
    --baseline-csv "$input_root/baseline_512.csv" \
    --fasta-file "$input_root/refseq_plastids_all.fna.gz" \
    --accession NC_053550.1 \
    --output-dir "$output_root" \
    --batch-size 4 \
    --temperature 0.8 \
    --seed 13 \
    --validate-only \
    2>&1 | tee "$output_root/logs/validation.log"

evaluation_started=$(date +%s)
/usr/bin/time -v -o "$output_root/logs/evaluation_resource_usage.txt" \
    python run_matched_context_eval.py \
        --checkpoint "$input_root/best_loss.pt" \
        --baseline-csv "$input_root/baseline_512.csv" \
        --fasta-file "$input_root/refseq_plastids_all.fna.gz" \
        --accession NC_053550.1 \
        --output-dir "$output_root" \
        --batch-size 4 \
        --temperature 0.8 \
        --seed 13 \
    2>&1 | tee "$output_root/logs/evaluation.log"
evaluation_finished=$(date +%s)
printf '%s\n' "$((evaluation_finished - evaluation_started))" > "$output_root/evaluation_runtime_seconds.txt"

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""
find "$output_root" -type f ! -name sha256sums.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$output_root/sha256sums.txt"
