import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.baselines.checkpoint import load_baseline_checkpoint
from src.baselines.frequency import generate_frequency
from src.baselines.markov import generate_markov
from src.dna.checkpoint import load_model
from src.evaluation.bootstrap import bootstrap_model_comparisons
from src.evaluation.test_panel import file_sha256, sequence_sha256
from src.evaluation.metrics import (
    add_alignment_identities,
    baseline_teacher_metrics,
    distance_bin_rows,
    neural_teacher_metrics,
    recursive_triplet_accuracy,
)
from src.sliding_eval.fasta import PlastidRecord, stream_fasta
from src.sliding_eval.generation import (
    exact_identity_percent,
    gc_difference_percent,
    generate_windows,
    longest_homopolymer_run,
)
from src.sliding_eval.regions import Region, RegionMap, infer_regions
from src.sliding_eval.windows import SlidingWindow, build_windows, write_windows_csv


LOCK_VERSION = 2
BASELINE_METHODS = {
    "most-common-base",
    "triplet-frequency",
    "markov",
}


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"test manifest contains no records: {path}")
    return rows


def load_panel_records(panel_config: dict) -> list[PlastidRecord]:
    manifest_path = Path(panel_config["manifest"])
    expected_hash = panel_config.get("frozen_manifest_sha256")
    actual_hash = file_sha256(manifest_path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"test manifest hash changed: expected {expected_hash}, found {actual_hash}"
        )

    records = []
    download_dir = Path(panel_config.get("download_dir", ""))
    for row in read_manifest(manifest_path):
        accession = row["accession"]
        source_path = download_dir / f"{accession}.fasta"
        if not source_path.exists() and row.get("source_type") == "external-fasta":
            source_path = Path(row["source"])
        candidates = [record for record in stream_fasta(str(source_path))]
        matches = [record for record in candidates if record.accession == accession]
        if len(matches) != 1:
            raise ValueError(f"expected one local FASTA record for {accession}: {source_path}")
        record = matches[0]
        if len(record.sequence) != int(row["length"]):
            raise ValueError(f"sequence length changed for {accession}")
        if sequence_sha256(record.sequence) != row["sequence_sha256"]:
            raise ValueError(f"sequence hash changed for {accession}")
        if set(record.sequence) - set("ACGT"):
            raise ValueError(f"test sequence contains non-ACGT bases: {accession}")
        records.append(record)
    return records


def validate_models_config(config: dict, require_frozen: bool) -> None:
    if require_frozen and config.get("frozen") is not True:
        raise ValueError("set frozen: true only after every final checkpoint is selected")
    settings = config.get("evaluation", {})
    required_settings = {
        "prompt_length",
        "generation_length",
        "stride",
        "circular",
        "decoding_mode",
        "temperature",
        "seed",
        "batch_size",
        "context_lengths",
        "context_targets_per_genome",
        "sampling_seeds",
        "baseline_smoothing",
        "edge_bases",
        "distance_bin_width",
        "alignment_scores",
        "statistical_baseline",
        "bootstrap_replicates",
        "bootstrap_confidence",
        "bootstrap_seed",
    }
    missing = sorted(required_settings - set(settings))
    if missing:
        raise ValueError(f"missing evaluation settings: {', '.join(missing)}")
    if settings["decoding_mode"] not in {"raw_greedy", "sampled"}:
        raise ValueError("decoding_mode must be raw_greedy or sampled")
    context_lengths = settings["context_lengths"]
    if (
        not context_lengths
        or len(context_lengths) != len(set(context_lengths))
        or any(not isinstance(length, int) or length <= 0 for length in context_lengths)
    ):
        raise ValueError("context_lengths must contain unique positive integers")
    if int(settings["context_targets_per_genome"]) <= 0:
        raise ValueError("context_targets_per_genome must be positive")
    seeds = settings["sampling_seeds"]
    if len(set(seeds)) < 2 or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("sampling_seeds must contain at least two unique integers")
    if int(settings["seed"]) not in seeds:
        raise ValueError("the primary seed must be included in sampling_seeds")
    if float(settings["baseline_smoothing"]) <= 0:
        raise ValueError("baseline_smoothing must be positive")
    if int(settings["edge_bases"]) <= 0:
        raise ValueError("edge_bases must be positive")
    if int(settings["generation_length"]) < 3:
        raise ValueError("triplet evaluation requires generation_length of at least three")
    if int(settings["edge_bases"]) > int(settings["generation_length"]):
        raise ValueError("edge_bases cannot exceed generation_length")
    if int(settings["distance_bin_width"]) <= 0:
        raise ValueError("distance_bin_width must be positive")
    alignment_scores = settings["alignment_scores"]
    if set(alignment_scores) != {"match", "mismatch", "gap_open", "gap_extension"}:
        raise ValueError("alignment_scores has missing or unknown fields")
    if int(settings["bootstrap_replicates"]) <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < float(settings["bootstrap_confidence"]) < 1:
        raise ValueError("bootstrap_confidence must be between zero and one")
    names = set()
    for model in config.get("models", []):
        name = model.get("name", "")
        if not name or name in names or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name):
            raise ValueError(f"invalid or duplicate model name: {name!r}")
        names.add(name)
        if model.get("kind") not in {"baseline", "neural"}:
            raise ValueError(f"invalid model kind for {name}")
        if not model.get("checkpoint"):
            raise ValueError(f"missing checkpoint for {name}")
        if model["kind"] == "baseline":
            if model.get("method") not in BASELINE_METHODS:
                raise ValueError(f"invalid baseline method for {name}")
            if model["method"] == "markov" and model.get("order") not in {1, 3, 6}:
                raise ValueError(f"invalid Markov order for {name}")
    if not names:
        raise ValueError("models config contains no models")
    required_models = set(config.get("required_models", []))
    missing_models = sorted(required_models - names)
    if missing_models:
        raise ValueError(f"required models are missing: {', '.join(missing_models)}")
    if settings["statistical_baseline"] not in names:
        raise ValueError("statistical_baseline must name a configured model")


def build_record_windows(
    record: PlastidRecord,
    settings: dict,
    region_map: RegionMap | None = None,
) -> list[SlidingWindow]:
    if region_map is None:
        region_map = RegionMap([Region("unknown", 0, record.length)], "unknown")
    return build_windows(
        record,
        region_map,
        int(settings["prompt_length"]),
        int(settings["generation_length"]),
        int(settings["stride"]),
        circular=bool(settings["circular"]),
    )


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in dict.fromkeys(indices)]


def context_target_starts(record: PlastidRecord, settings: dict) -> list[int]:
    main_windows = build_record_windows(record, settings)
    candidates = [window.target_start for window in main_windows]
    return evenly_spaced(candidates, int(settings["context_targets_per_genome"]))


def build_context_windows(
    record: PlastidRecord,
    settings: dict,
    context_length: int,
    targets: list[int],
    region_map: RegionMap | None = None,
) -> list[SlidingWindow]:
    if region_map is None:
        region_map = RegionMap([Region("unknown", 0, record.length)], "unknown")
    starts = [(target - context_length) % record.length for target in targets]
    windows = build_windows(
        record,
        region_map,
        context_length,
        int(settings["generation_length"]),
        int(settings["stride"]),
        window_starts=starts,
        circular=bool(settings["circular"]),
    )
    if [window.target_start for window in windows] != targets:
        raise RuntimeError("context windows do not preserve target coordinates")
    return windows


def serialize_region_map(region_map: RegionMap) -> dict:
    return {
        "status": region_map.status,
        "regions": [
            {"name": region.name, "start": region.start, "end": region.end}
            for region in region_map.regions
        ],
    }


def deserialize_region_map(value: dict) -> RegionMap:
    return RegionMap(
        [Region(row["name"], int(row["start"]), int(row["end"])) for row in value["regions"]],
        value["status"],
    )


def resolve_region_map(record: PlastidRecord, panel_config: dict) -> RegionMap:
    annotation_dir = panel_config.get("annotation_dir")
    if annotation_dir:
        genbank_path = Path(annotation_dir) / f"{record.accession}.gb"
        if genbank_path.exists():
            from src.biological_eval.annotations import genbank_region_map

            region_map = genbank_region_map(genbank_path)
            if region_map is not None:
                return region_map
    return infer_regions(record.sequence)


def windows_fingerprint(
    records: list[PlastidRecord],
    settings: dict,
    region_maps: dict[str, RegionMap] | None = None,
) -> tuple[str, list[dict]]:
    digest = hashlib.sha256()
    rows = []
    for record in records:
        region_map = region_maps.get(record.accession) if region_maps else None
        for window in build_record_windows(record, settings, region_map):
            row = {
                "accession": record.accession,
                "window_start": window.window_start,
                "prompt_sha256": sequence_sha256(window.prompt),
                "true_suffix_sha256": sequence_sha256(window.true_suffix),
                "region": window.region,
                "region_source": window.region_source,
            }
            digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
            rows.append(row)
    return digest.hexdigest(), rows


def context_windows_fingerprint(
    records: list[PlastidRecord],
    settings: dict,
    region_maps: dict[str, RegionMap],
) -> tuple[str, list[dict]]:
    digest = hashlib.sha256()
    rows = []
    for record in records:
        targets = context_target_starts(record, settings)
        expected_truth: dict[int, str] = {}
        for context_length in settings["context_lengths"]:
            windows = build_context_windows(
                record,
                settings,
                int(context_length),
                targets,
                region_maps[record.accession],
            )
            for window in windows:
                truth_hash = sequence_sha256(window.true_suffix)
                old_hash = expected_truth.setdefault(window.target_start, truth_hash)
                if old_hash != truth_hash:
                    raise RuntimeError("context lengths produced different target suffixes")
                row = {
                    "accession": record.accession,
                    "target_start": window.target_start,
                    "context_length": int(context_length),
                    "prompt_sha256": sequence_sha256(window.prompt),
                    "true_suffix_sha256": truth_hash,
                    "region": window.region,
                    "region_source": window.region_source,
                }
                digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
                digest.update(b"\n")
                rows.append(row)
    return digest.hexdigest(), rows


def checkpoint_details(models: list[dict]) -> tuple[list[dict], str, str]:
    details = []
    train_fingerprint = ""
    validation_fingerprint = ""
    validation_window_fingerprint = ""
    loaded_paths = {}
    for model in models:
        checkpoint_path = Path(model["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found for {model['name']}: {checkpoint_path}")
        path_key = str(checkpoint_path.resolve())
        if path_key not in loaded_paths:
            if model["kind"] == "baseline":
                checkpoint = load_baseline_checkpoint(checkpoint_path)
                training = checkpoint["training"]
                metadata = {
                    "kind": "baseline",
                    "train_record_fingerprint": training["train_record_fingerprint"],
                    "validation_record_fingerprint": training["validation_record_fingerprint"],
                    "training_fasta_sha256": training["fasta_sha256"],
                }
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                prediction_unit = checkpoint.get(
                    "prediction_unit",
                    checkpoint.get("model_config", {}).get("prediction_unit", "base"),
                )
                if prediction_unit != "triplet":
                    raise ValueError(f"final comparison requires a triplet checkpoint: {model['name']}")
                fingerprints = checkpoint.get("data_fingerprints", {})
                metadata = {
                    "kind": "neural",
                    "model_type": checkpoint.get(
                        "model_type", checkpoint.get("model_config", {}).get("model_type", "s4d")
                    ),
                    "epoch": checkpoint.get("epoch"),
                    "train_record_fingerprint": fingerprints.get("train_records", ""),
                    "validation_record_fingerprint": fingerprints.get("val_records", ""),
                    "validation_window_fingerprint": fingerprints.get("val_windows", ""),
                }
            loaded_paths[path_key] = {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                **metadata,
            }
        detail = {"name": model["name"], **loaded_paths[path_key]}
        details.append(detail)
        current_train = detail.get("train_record_fingerprint", "")
        current_val = detail.get("validation_record_fingerprint", "")
        if not current_train or not current_val:
            raise ValueError(f"checkpoint lacks data fingerprints: {model['name']}")
        if train_fingerprint and current_train != train_fingerprint:
            raise ValueError("model training-record fingerprints do not match")
        if validation_fingerprint and current_val != validation_fingerprint:
            raise ValueError("model validation-record fingerprints do not match")
        train_fingerprint = current_train
        validation_fingerprint = current_val
        current_val_windows = detail.get("validation_window_fingerprint", "")
        if current_val_windows:
            if (
                validation_window_fingerprint
                and current_val_windows != validation_window_fingerprint
            ):
                raise ValueError("neural validation-window fingerprints do not match")
            validation_window_fingerprint = current_val_windows
    return details, train_fingerprint, validation_fingerprint


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_result_csv(record: PlastidRecord, windows: list[SlidingWindow], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".writing"
    temporary_dir.mkdir(exist_ok=True)
    temporary_path = Path(write_windows_csv(record, windows, str(temporary_dir)))
    final_path = output_dir / temporary_path.name
    temporary_path.replace(final_path)
    temporary_dir.rmdir()
    return final_path


def freeze_evaluation(
    panel_config_path: str,
    models_config_path: str,
    output_dir: str,
) -> Path:
    panel_config = load_yaml(panel_config_path)
    if panel_config.get("panel_role") != "untouched-test":
        raise ValueError("freeze requires a panel labeled untouched-test")
    if not panel_config.get("frozen_manifest_sha256"):
        raise ValueError("untouched panel must provide frozen_manifest_sha256")
    models_config = load_yaml(models_config_path)
    validate_models_config(models_config, require_frozen=True)
    output = Path(output_dir)
    lock_path = output / "evaluation_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"evaluation is already frozen: {lock_path}")

    records = load_panel_records(panel_config)
    region_maps = {
        record.accession: resolve_region_map(record, panel_config) for record in records
    }
    window_hash, window_rows = windows_fingerprint(
        records,
        models_config["evaluation"],
        region_maps,
    )
    context_hash, context_rows = context_windows_fingerprint(
        records,
        models_config["evaluation"],
        region_maps,
    )
    details, train_fingerprint, val_fingerprint = checkpoint_details(models_config["models"])
    baseline_details = [row for row in details if row["kind"] == "baseline"]
    metadata_path = Path(panel_config["metadata"])
    panel_metadata = json.loads(metadata_path.read_text())
    for detail in baseline_details:
        if detail["training_fasta_sha256"] != panel_metadata["training_fasta_sha256"]:
            raise ValueError("baseline and untouched panel use different training FASTA files")

    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "frozen_windows.csv", window_rows)
    write_rows(output / "frozen_context_windows.csv", context_rows)
    lock = {
        "lock_version": LOCK_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_config": str(panel_config_path),
        "panel_config_sha256": file_sha256(panel_config_path),
        "manifest": panel_config["manifest"],
        "manifest_sha256": file_sha256(panel_config["manifest"]),
        "panel_metadata": panel_config["metadata"],
        "panel_metadata_sha256": file_sha256(panel_config["metadata"]),
        "models_config": str(models_config_path),
        "models_config_sha256": file_sha256(models_config_path),
        "evaluation": models_config["evaluation"],
        "checkpoints": details,
        "train_record_fingerprint": train_fingerprint,
        "validation_record_fingerprint": val_fingerprint,
        "accessions": [record.accession for record in records],
        "windows": len(window_rows),
        "windows_sha256": window_hash,
        "context_windows": len(context_rows),
        "context_windows_sha256": context_hash,
        "region_maps": {
            accession: serialize_region_map(region_map)
            for accession, region_map in region_maps.items()
        },
    }
    write_json(lock_path, lock)
    return lock_path


def verify_evaluation_lock(
    panel_config_path: str,
    models_config_path: str,
    output_dir: str,
) -> tuple[dict, dict, list[PlastidRecord]]:
    lock_path = Path(output_dir) / "evaluation_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("evaluation is not frozen; run evaluate_models.py with --freeze first")
    lock = json.loads(lock_path.read_text())
    if lock.get("lock_version") != LOCK_VERSION:
        raise ValueError("unsupported evaluation lock version")
    expected_files = {
        panel_config_path: lock["panel_config_sha256"],
        models_config_path: lock["models_config_sha256"],
        lock["manifest"]: lock["manifest_sha256"],
        lock["panel_metadata"]: lock["panel_metadata_sha256"],
    }
    for path, expected in expected_files.items():
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen file changed: {path}")
    for detail in lock["checkpoints"]:
        if file_sha256(detail["path"]) != detail["sha256"]:
            raise ValueError(f"frozen checkpoint changed: {detail['path']}")

    panel_config = load_yaml(panel_config_path)
    models_config = load_yaml(models_config_path)
    validate_models_config(models_config, require_frozen=True)
    records = load_panel_records(panel_config)
    region_maps = {
        accession: deserialize_region_map(value)
        for accession, value in lock["region_maps"].items()
    }
    window_hash, window_rows = windows_fingerprint(
        records,
        models_config["evaluation"],
        region_maps,
    )
    if window_hash != lock["windows_sha256"] or len(window_rows) != lock["windows"]:
        raise ValueError("frozen evaluation windows changed")
    context_hash, context_rows = context_windows_fingerprint(
        records,
        models_config["evaluation"],
        region_maps,
    )
    if (
        context_hash != lock["context_windows_sha256"]
        or len(context_rows) != lock["context_windows"]
    ):
        raise ValueError("frozen context windows changed")
    return lock, models_config, records


def generate_baseline_windows(
    windows: list[SlidingWindow],
    model_config: dict,
    checkpoint: dict,
) -> None:
    method = model_config["method"]
    for window in windows:
        if method == "markov":
            generated = generate_markov(
                window.prompt,
                len(window.true_suffix),
                int(model_config["order"]),
                checkpoint["markov_counts"],
                checkpoint["triplet_counts"],
                checkpoint["triplet_vocabulary"],
            )
        else:
            generated = generate_frequency(
                method,
                len(window.true_suffix),
                checkpoint["base_counts"],
                checkpoint["triplet_counts"],
                checkpoint["triplet_vocabulary"],
            )
        window.generated_suffix = generated
        window.generated_length = len(generated)
        window.accuracy_percent = exact_identity_percent(generated, window.true_suffix)
        window.decoding_mode = "deterministic-count-baseline"
        window.longest_generated_run = longest_homopolymer_run(generated)
        window.n_count = generated.count("N")
        window.gc_difference_percent = gc_difference_percent(generated, window.true_suffix)


def validate_result_csv(path: Path, expected: list[SlidingWindow]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError(f"existing result has the wrong row count: {path}")
    for index, (row, window) in enumerate(zip(rows, expected)):
        if int(row["window_start"]) != window.window_start:
            raise ValueError(f"window start changed in {path} row {index}")
        if row["prompt"] != window.prompt or row["true_suffix"] != window.true_suffix:
            raise ValueError(f"frozen sequence changed in {path} row {index}")
        generated = row["generated_suffix"]
        if len(generated) != len(window.true_suffix) or set(generated) - set("ACGT"):
            raise ValueError(f"invalid generated sequence in {path} row {index}")
    return rows


def summarize_rows(rows: list[dict[str, str]], edge_bases: int = 100) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty results")
    generated_length = len(rows[0]["true_suffix"])
    total_matches = first_matches = final_matches = 0
    longest_run = invalid_bases = rows_over_20 = 0
    for row in rows:
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        total_matches += sum(a == b for a, b in zip(generated, truth))
        first_length = min(edge_bases, len(truth))
        final_length = min(edge_bases, len(truth))
        first_matches += sum(a == b for a, b in zip(generated[:first_length], truth[:first_length]))
        final_matches += sum(a == b for a, b in zip(generated[-final_length:], truth[-final_length:]))
        run = longest_homopolymer_run(generated)
        longest_run = max(longest_run, run)
        rows_over_20 += int(run > 20)
        invalid_bases += sum(base not in "ACGT" for base in generated)
    count = len(rows)
    edge_length = min(edge_bases, generated_length)
    triplet_correct, triplet_total = recursive_triplet_accuracy(rows)
    summary = {
        "windows": count,
        "bases": count * generated_length,
        "accuracy_percent": 100 * total_matches / (count * generated_length),
        "first_100_accuracy_percent": 100 * first_matches / (count * edge_length),
        "final_100_accuracy_percent": 100 * final_matches / (count * edge_length),
        "longest_repeated_base_run": longest_run,
        "rows_with_runs_over_20": rows_over_20,
        "invalid_generated_bases": invalid_bases,
        "recursive_exact_triplet_accuracy_percent": 100 * triplet_correct / triplet_total,
    }
    if all("alignment_identity_percent" in row for row in rows):
        summary["mean_global_alignment_identity_percent"] = sum(
            float(row["alignment_identity_percent"]) for row in rows
        ) / count
    return summary


def summarize_groups(
    rows: list[dict[str, str]],
    keys: list[str],
    edge_bases: int = 100,
) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                **dict(zip(keys, key)),
                **summarize_rows(group_rows, edge_bases),
            }
        )
    return summaries


def panel_record_metadata(panel_config: dict) -> dict[str, dict[str, str]]:
    groups = panel_config.get("plant_groups", {})
    accession_groups = {
        accession: group
        for group, accessions in groups.items()
        for accession in accessions
    }
    metadata = {}
    for row in read_manifest(panel_config["manifest"]):
        accession = row["accession"]
        metadata[accession] = {
            "accession": accession,
            "species": row.get("species", ""),
            "genus": row.get("genus", ""),
            "plant_group": accession_groups.get(
                accession,
                row.get("plant_group") or row.get("genus") or "unknown",
            ),
        }
    return metadata


def generate_model_windows(
    windows: list[SlidingWindow],
    model_config: dict,
    settings: dict,
    baseline_checkpoint: dict | None,
    seed: int,
    model_bundle: tuple | None = None,
) -> None:
    if model_config["kind"] == "baseline":
        generate_baseline_windows(windows, model_config, baseline_checkpoint)
    else:
        generate_windows(
            windows,
            checkpoint=model_config["checkpoint"],
            generate_length=int(settings["generation_length"]),
            batch_size=int(settings["batch_size"]),
            seed=seed,
            decoding_mode=settings["decoding_mode"],
            temperature=float(settings["temperature"]),
            model_bundle=model_bundle,
        )


def run_shared_evaluation(
    panel_config_path: str,
    models_config_path: str,
    output_dir: str,
) -> Path:
    complete_path = Path(output_dir) / "run_complete.json"
    if complete_path.exists():
        raise FileExistsError(
            f"untouched evaluation is already complete and must not be rerun: {complete_path}"
        )
    lock, models_config, records = verify_evaluation_lock(
        panel_config_path,
        models_config_path,
        output_dir,
    )
    panel_config = load_yaml(panel_config_path)
    record_metadata = panel_record_metadata(panel_config)
    region_maps = {
        accession: deserialize_region_map(value)
        for accession, value in lock["region_maps"].items()
    }
    settings = models_config["evaluation"]
    edge_bases = int(settings["edge_bases"])
    alignment_scores = {
        key: float(value) for key, value in settings["alignment_scores"].items()
    }
    summary_rows = []
    genome_rows = []
    teacher_rows = []
    plant_group_source_rows: list[dict[str, str]] = []
    region_source_rows: list[dict[str, str]] = []
    distance_source_rows: list[dict[str, str]] = []
    context_rows = []
    context_genome_rows = []
    teacher_windows = [
        window
        for record in records
        for window in build_record_windows(record, settings, region_maps[record.accession])
    ]
    for model_config in models_config["models"]:
        baseline_checkpoint = None
        model_bundle = None
        if model_config["kind"] == "baseline":
            baseline_checkpoint = load_baseline_checkpoint(model_config["checkpoint"])
            teacher = baseline_teacher_metrics(
                model_config,
                baseline_checkpoint,
                teacher_windows,
                float(settings["baseline_smoothing"]),
            )
            seed_pairs = [("deterministic", int(settings["seed"]))]
        else:
            model_bundle = load_model(model_config["checkpoint"])
            teacher = neural_teacher_metrics(
                model_config["checkpoint"],
                teacher_windows,
                int(settings["batch_size"]),
                model_bundle=model_bundle,
            )
            seed_pairs = [(str(seed), int(seed)) for seed in settings["sampling_seeds"]]
        teacher_rows.append(
            {
                "model": model_config["name"],
                "kind": model_config["kind"],
                **teacher,
            }
        )

        for seed_label, seed in seed_pairs:
            all_rows = []
            model_dir = Path(output_dir) / "models" / model_config["name"] / f"seed_{seed_label}"
            for record in records:
                windows = build_record_windows(
                    record,
                    settings,
                    region_maps[record.accession],
                )
                csv_path = model_dir / f"{record.accession}_windows.csv"
                if csv_path.exists():
                    rows = validate_result_csv(csv_path, windows)
                else:
                    generate_model_windows(
                        windows,
                        model_config,
                        settings,
                        baseline_checkpoint,
                        seed,
                        model_bundle,
                    )
                    write_result_csv(record, windows, model_dir)
                    rows = validate_result_csv(csv_path, windows)
                add_alignment_identities(rows, alignment_scores)
                all_rows.extend(rows)
                metadata = record_metadata[record.accession]
                genome_rows.append(
                    {
                        "model": model_config["name"],
                        "kind": model_config["kind"],
                        "seed": seed_label,
                        **metadata,
                        **summarize_rows(rows, edge_bases),
                    }
                )
                for row in rows:
                    shared = {
                        **row,
                        "model": model_config["name"],
                        "kind": model_config["kind"],
                        "seed": seed_label,
                    }
                    plant_group_source_rows.append(
                        {**shared, "plant_group": metadata["plant_group"]}
                    )
                    region_source_rows.append(shared)
                    distance_source_rows.append(shared)
            summary_rows.append(
                {
                    "model": model_config["name"],
                    "kind": model_config["kind"],
                    "seed": seed_label,
                    **summarize_rows(all_rows, edge_bases),
                }
            )
            write_rows(model_dir / "summary.csv", [summary_rows[-1]])

            model_context_rows: list[dict[str, str]] = []
            for record in records:
                targets = context_target_starts(record, settings)
                for context_length in settings["context_lengths"]:
                    windows = build_context_windows(
                        record,
                        settings,
                        int(context_length),
                        targets,
                        region_maps[record.accession],
                    )
                    context_dir = (
                        Path(output_dir)
                        / "context"
                        / model_config["name"]
                        / f"seed_{seed_label}"
                        / f"context_{context_length}"
                    )
                    csv_path = context_dir / f"{record.accession}_windows.csv"
                    if csv_path.exists():
                        rows = validate_result_csv(csv_path, windows)
                    else:
                        generate_model_windows(
                            windows,
                            model_config,
                            settings,
                            baseline_checkpoint,
                            seed,
                            model_bundle,
                        )
                        write_result_csv(record, windows, context_dir)
                        rows = validate_result_csv(csv_path, windows)
                    metadata = record_metadata[record.accession]
                    context_genome_rows.append(
                        {
                            "model": model_config["name"],
                            "kind": model_config["kind"],
                            "seed": seed_label,
                            "context_length": int(context_length),
                            **metadata,
                            **summarize_rows(rows, edge_bases),
                        }
                    )
                    for row in rows:
                        model_context_rows.append(
                            {**row, "context_length": str(context_length)}
                        )
            for row in summarize_groups(
                model_context_rows,
                ["context_length"],
                edge_bases,
            ):
                context_rows.append(
                    {
                        "model": model_config["name"],
                        "kind": model_config["kind"],
                        "seed": seed_label,
                        **row,
                    }
                )

    summary_path = Path(output_dir) / "model_comparison.csv"
    write_rows(summary_path, summary_rows)
    write_rows(Path(output_dir) / "teacher_forced_metrics.csv", teacher_rows)
    write_rows(Path(output_dir) / "results_by_genome.csv", genome_rows)
    write_rows(
        Path(output_dir) / "results_by_plant_group.csv",
        summarize_groups(
            plant_group_source_rows,
            ["model", "kind", "seed", "plant_group"],
            edge_bases,
        ),
    )
    write_rows(
        Path(output_dir) / "results_by_region.csv",
        summarize_groups(
            region_source_rows,
            ["model", "kind", "seed", "region", "region_source"],
            edge_bases,
        ),
    )
    write_rows(
        Path(output_dir) / "accuracy_by_distance.csv",
        distance_bin_rows(
            distance_source_rows,
            ["model", "kind", "seed"],
            int(settings["distance_bin_width"]),
        ),
    )
    write_rows(
        Path(output_dir) / "accuracy_by_distance_and_genome.csv",
        distance_bin_rows(
            distance_source_rows,
            ["model", "kind", "seed", "accession"],
            int(settings["distance_bin_width"]),
        ),
    )
    write_rows(Path(output_dir) / "context_length_comparison.csv", context_rows)
    write_rows(Path(output_dir) / "context_length_by_genome.csv", context_genome_rows)
    averaged_genomes, model_statistics, paired_statistics, baseline_statistics = (
        bootstrap_model_comparisons(
            genome_rows,
            settings["statistical_baseline"],
            int(settings["bootstrap_replicates"]),
            float(settings["bootstrap_confidence"]),
            int(settings["bootstrap_seed"]),
        )
    )
    write_rows(Path(output_dir) / "statistics_genome_seed_averages.csv", averaged_genomes)
    write_rows(Path(output_dir) / "statistics_model_accuracy.csv", model_statistics)
    write_rows(Path(output_dir) / "statistics_paired_differences.csv", paired_statistics)
    write_rows(Path(output_dir) / "statistics_vs_baseline.csv", baseline_statistics)
    write_json(
        complete_path,
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": len(models_config["models"]),
            "recursive_runs": len(summary_rows),
            "summary": str(summary_path),
            "alignment": {
                "mode": "global",
                "identity_denominator": "matches + mismatches + gap columns",
                "scores": alignment_scores,
            },
            "edge_bases": edge_bases,
            "distance_bin_width": int(settings["distance_bin_width"]),
            "sampling_seeds": settings["sampling_seeds"],
            "bootstrap": {
                "unit": "whole genome",
                "seed_aggregation": "mean within each genome before resampling",
                "baseline": settings["statistical_baseline"],
                "replicates": int(settings["bootstrap_replicates"]),
                "confidence": float(settings["bootstrap_confidence"]),
                "seed": int(settings["bootstrap_seed"]),
            },
        },
    )
    return summary_path
