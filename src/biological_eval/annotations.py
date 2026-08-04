import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio import Entrez, SeqIO
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from src.biological_eval.config import require_keys


Entrez.email = "guhanst9@gmail.com"


@dataclass(frozen=True)
class FeatureRecord:
    accession: str
    feature_type: str
    gene: str
    strand: int
    start: int
    end: int


def fetch_genbank(accession: str, cache_dir: str, overwrite: bool = False) -> Path:
    path = Path(cache_dir) / f"{accession}.gb"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    with Entrez.efetch(db="nuccore", id=accession, rettype="gbwithparts", retmode="text") as handle:
        text = handle.read()
    if "LOCUS" not in text:
        raise ValueError(f"NCBI did not return GenBank data for {accession}")
    path.write_text(text)
    return path


def feature_gene(feature: SeqFeature) -> str:
    for key in ("gene", "product", "locus_tag"):
        values = feature.qualifiers.get(key)
        if values:
            return values[0]
    return ""


def feature_intervals(feature: SeqFeature) -> list[tuple[int, int]]:
    location = feature.location
    if isinstance(location, CompoundLocation):
        return [(int(part.start), int(part.end)) for part in location.parts]
    if isinstance(location, FeatureLocation):
        return [(int(location.start), int(location.end))]
    return []


def parse_genbank_features(path: str | Path) -> tuple[str, int, list[FeatureRecord]]:
    record = SeqIO.read(str(path), "genbank")
    accession = record.id.split()[0]
    features: list[FeatureRecord] = []
    for feature in record.features:
        if feature.type not in {"CDS", "tRNA", "rRNA", "repeat_region"}:
            continue
        strand = int(feature.location.strand or 0)
        gene = feature_gene(feature)
        for start, end in feature_intervals(feature):
            features.append(
                FeatureRecord(
                    accession=accession,
                    feature_type=feature.type,
                    gene=gene,
                    strand=strand,
                    start=start,
                    end=end,
                )
            )
    return accession, len(record.seq), features


def split_circular_interval(start: int, end: int, genome_length: int) -> list[tuple[int, int]]:
    start %= genome_length
    end %= genome_length
    if start < end:
        return [(start, end)]
    if start == end:
        return [(0, genome_length)]
    return [(start, genome_length), (0, end)]


def interval_overlap(start: int, end: int, feature_start: int, feature_end: int, genome_length: int) -> int:
    overlap = 0
    for q_start, q_end in split_circular_interval(start, end, genome_length):
        for f_start, f_end in split_circular_interval(feature_start, feature_end, genome_length):
            overlap += max(0, min(q_end, f_end) - max(q_start, f_start))
    return overlap


def write_features_csv(path: Path, rows: list[FeatureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["accession", "feature_type", "gene", "strand", "start", "end"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def selected_accessions(config: dict[str, Any], max_genomes: int | None) -> list[str]:
    accessions = [item["accession"] for item in config["panel"]]
    return accessions if max_genomes is None else accessions[:max_genomes]


def run_annotations(config: dict[str, Any], output_dir: str, max_genomes: int | None, overwrite: bool) -> None:
    require_keys(config, ["panel"])
    cache_dir = str(Path(output_dir) / "annotations")
    all_features: list[FeatureRecord] = []
    accessions = selected_accessions(config, max_genomes)
    for accession in accessions:
        path = fetch_genbank(accession, cache_dir, overwrite=overwrite)
        parsed_accession, length, features = parse_genbank_features(path)
        all_features.extend(features)
        print(f"  {parsed_accession}: length={length} features={len(features)}")
    output_path = Path(output_dir) / "annotation_features.csv"
    write_features_csv(output_path, all_features)
    print("Annotation stage complete")
    print(f"  Accessions: {len(accessions)}")
    print(f"  Features: {len(all_features)}")
    print(f"  Feature CSV: {output_path}")
