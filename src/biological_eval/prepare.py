import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from src.biological_eval.config import require_keys
from src.sliding_eval.fasta import find_record_by_accession


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict", {})
    parameter_count = sum(tensor.numel() for tensor in model_state.values())
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_size_bytes": os.path.getsize(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "git_commit": current_git_commit(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "objective": checkpoint.get("objective"),
        "data_type": checkpoint.get("data_type"),
        "preset": checkpoint.get("preset"),
        "holdout_accession": checkpoint.get("holdout_accession"),
        "parameter_count": parameter_count,
        "model_config": checkpoint.get("model_config", {}),
        "tokenizer_vocab": checkpoint.get("tokenizer_vocab", {}),
        "preset_config": checkpoint.get("preset_config", {}),
        "recovery_settings": checkpoint.get("recovery_settings", {}),
        "free_generation_metrics": checkpoint.get("free_generation_metrics", {}),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_quality_score": checkpoint.get("best_quality_score"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_panel_manifest(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_fasta = config["raw_fasta"]
    fieldnames = [
        "accession",
        "name",
        "group",
        "exposure",
        "present_in_raw_fasta",
        "header",
        "length",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in config["panel"]:
            accession = item["accession"]
            try:
                record = find_record_by_accession(raw_fasta, accession)
                present = "yes"
                header = record.header
                length = record.length
            except ValueError:
                present = "no"
                header = ""
                length = ""
            writer.writerow(
                {
                    "accession": accession,
                    "name": item["name"],
                    "group": item["group"],
                    "exposure": item["exposure"],
                    "present_in_raw_fasta": present,
                    "header": header,
                    "length": length,
                }
            )


def run_prepare(config: dict[str, Any], output_dir: str) -> None:
    require_keys(
        config,
        [
            "checkpoint",
            "raw_fasta",
            "clean_fasta",
            "holdout_accession",
            "panel",
            "known_rosa_baseline",
        ],
    )
    output_root = Path(output_dir)
    metadata = checkpoint_metadata(config["checkpoint"])
    metadata["config_path"] = config.get("_config_path")
    metadata["raw_fasta"] = config["raw_fasta"]
    metadata["clean_fasta"] = config["clean_fasta"]
    metadata["known_rosa_baseline"] = config["known_rosa_baseline"]
    write_json(output_root / "baseline_metadata.json", metadata)
    write_panel_manifest(config, output_root / "panel_manifest.csv")
    print("Prepare stage complete")
    print(f"  Metadata: {output_root / 'baseline_metadata.json'}")
    print(f"  Panel manifest: {output_root / 'panel_manifest.csv'}")
    print(f"  Checkpoint SHA-256: {metadata['checkpoint_sha256']}")

