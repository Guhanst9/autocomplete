import csv
from pathlib import Path
from typing import Any

from src.biological_eval.annotations import genbank_region_map
from src.biological_eval.config import require_keys
from src.sliding_eval.fasta import PlastidRecord, stream_fasta
from src.sliding_eval.generation import generate_windows
from src.sliding_eval.regions import RegionMap, infer_regions, label_interval
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


def load_region_map(accession: str, sequence: str, output_dir: str) -> RegionMap:
    genbank_path = Path(output_dir) / "annotations" / f"{accession}.gb"
    if genbank_path.exists():
        region_map = genbank_region_map(genbank_path)
        if region_map is not None:
            return region_map
    return infer_regions(sequence)


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

    for item in panel:
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

        region_map = load_region_map(accession, record.sequence, output_dir)
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


def run_relabel(config: dict[str, Any], output_dir: str, max_genomes: int | None) -> None:
    require_keys(config, ["raw_fasta", "panel"])
    panel = selected_panel(config, max_genomes)
    records = load_panel_records(config["raw_fasta"], panel)
    maps = {
        accession: load_region_map(accession, record.sequence, output_dir)
        for accession, record in records.items()
    }
    changed = 0
    row_count = 0
    csv_count = 0
    for csv_path in sorted(Path(output_dir).glob("sliding*/*/*/*_windows.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not rows:
            continue
        accession = rows[0]["accession"]
        record = records.get(accession)
        region_map = maps.get(accession)
        if record is None or region_map is None:
            continue
        if "region_source" not in fieldnames:
            fieldnames.insert(fieldnames.index("region") + 1, "region_source")
        for row in rows:
            target_start = int(row["target_start"])
            target_length = len(row["true_suffix"])
            new_region = label_interval(
                target_start,
                target_start + target_length,
                region_map,
                record.length,
            )
            changed += new_region != row["region"]
            row["region"] = new_region
            row["region_source"] = region_map.status
            row_count += 1
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
        csv_count += 1

    print("Relabel stage complete")
    print(f"  CSV files: {csv_count}")
    print(f"  Rows: {row_count}")
    print(f"  Changed labels: {changed}")
