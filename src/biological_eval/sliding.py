import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.biological_eval.config import require_keys
from src.sliding_eval.fasta import PlastidRecord, stream_fasta
from src.sliding_eval.generation import generate_windows
from src.sliding_eval.regions import infer_regions
from src.sliding_eval.windows import build_windows, write_windows_csv


def parse_seed_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def selected_panel(config: dict[str, Any], max_genomes: int | None) -> list[dict[str, Any]]:
    panel = list(config["panel"])
    if max_genomes is not None:
        panel = panel[:max_genomes]
    return panel


def load_panel_records(fasta_file: str, panel: list[dict[str, Any]]) -> dict[str, PlastidRecord]:
    wanted = {item["accession"] for item in panel}
    records: dict[str, PlastidRecord] = {}
    for record in stream_fasta(fasta_file):
        if record.accession in wanted:
            records[record.accession] = record
            if len(records) == len(wanted):
                break
    return records


def temperature_label(temperature: float) -> str:
    text = f"{temperature:.2f}".rstrip("0").rstrip(".")
    return "t" + text.replace(".", "")


def sliding_run_name(max_genomes: int | None, max_windows: int | None) -> str:
    if max_genomes is None and max_windows is None:
        return "sliding"
    genome_part = "all" if max_genomes is None else str(max_genomes)
    window_part = "all" if max_windows is None else str(max_windows)
    return f"sliding_smoke_g{genome_part}_w{window_part}"


def run_output_dir(
    output_dir: str,
    accession: str,
    decoding_mode: str,
    temperature: float,
    seed: int,
    max_genomes: int | None,
    max_windows: int | None,
) -> Path:
    if decoding_mode == "sampled":
        decode_label = f"sampled_{temperature_label(temperature)}_seed{seed}"
    else:
        decode_label = f"{decoding_mode}_seed{seed}"
    return Path(output_dir) / sliding_run_name(max_genomes, max_windows) / accession / decode_label


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "accession",
        "group",
        "exposure",
        "decoding_mode",
        "temperature",
        "seed",
        "status",
        "windows",
        "csv_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sliding(
    config: dict[str, Any],
    output_dir: str,
    max_genomes: int | None,
    max_windows: int | None,
    seeds_value: str | None,
    overwrite: bool,
) -> None:
    require_keys(
        config,
        [
            "raw_fasta",
            "checkpoint",
            "panel",
            "prompt_length",
            "generation_length",
            "stride",
            "circular",
            "primary_decoding",
        ],
    )
    decoding = config["primary_decoding"]
    decoding_mode = decoding["mode"]
    temperature = float(decoding.get("temperature", 1.0))
    default_seeds = [int(decoding.get("seed", 13))]
    seeds = parse_seed_list(seeds_value, default_seeds)

    panel = selected_panel(config, max_genomes)
    records = load_panel_records(config["raw_fasta"], panel)
    manifest_rows: list[dict[str, Any]] = []

    for item in tqdm(panel, desc="Genomes"):
        accession = item["accession"]
        record = records.get(accession)
        if record is None:
            manifest_rows.append(
                {
                    "accession": accession,
                    "group": item["group"],
                    "exposure": item["exposure"],
                    "decoding_mode": decoding_mode,
                    "temperature": temperature,
                    "seed": "",
                    "status": "missing-local-fasta",
                    "windows": 0,
                    "csv_path": "",
                }
            )
            continue

        region_map = infer_regions(record.sequence)
        for seed in seeds:
            target_dir = run_output_dir(
                output_dir,
                accession,
                decoding_mode,
                temperature,
                seed,
                max_genomes,
                max_windows,
            )
            csv_path = target_dir / f"{accession}_windows.csv"
            if csv_path.exists() and not overwrite:
                status = "skipped-existing"
                window_count = ""
                tqdm.write(f"{accession} seed {seed}: skipped existing CSV")
            else:
                windows = build_windows(
                    record,
                    region_map,
                    config["prompt_length"],
                    config["generation_length"],
                    config["stride"],
                    max_windows=max_windows,
                    circular=bool(config["circular"]),
                )
                tqdm.write(
                    f"{accession} seed {seed}: generating {len(windows)} windows "
                    f"({decoding_mode}, temperature={temperature})"
                )
                generate_windows(
                    windows,
                    checkpoint=config["checkpoint"],
                    generate_length=config["generation_length"],
                    batch_size=4,
                    seed=seed,
                    decoding_mode=decoding_mode,
                    temperature=temperature,
                )
                write_windows_csv(record, windows, str(target_dir))
                status = "generated"
                window_count = len(windows)

            manifest_rows.append(
                {
                    "accession": accession,
                    "group": item["group"],
                    "exposure": item["exposure"],
                    "decoding_mode": decoding_mode,
                    "temperature": temperature,
                    "seed": seed,
                    "status": status,
                    "windows": window_count,
                    "csv_path": csv_path,
                }
            )

    manifest_path = Path(output_dir) / sliding_run_name(max_genomes, max_windows) / "sliding_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print("Sliding stage complete")
    print(f"  Panel genomes requested: {len(panel)}")
    print(f"  Seeds: {','.join(str(seed) for seed in seeds)}")
    print(f"  Manifest: {manifest_path}")
