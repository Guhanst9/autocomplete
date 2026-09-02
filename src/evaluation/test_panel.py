import csv
import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from Bio import Entrez


IUPAC_DNA = set("ACGTRYSWKMBDHVN")
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
ACCESSION_PATTERN = re.compile(r"^[A-Z]{1,6}_?[A-Z0-9]+(?:\.\d+)?$")


@dataclass(frozen=True)
class PanelRecord:
    accession: str
    header: str
    sequence: str
    source: str
    source_type: str


@dataclass
class TrainingIndex:
    accessions: dict[str, str]
    sequence_hashes: dict[str, str]
    genera: set[str]
    records: int


@dataclass(frozen=True)
class Rejection:
    accession: str
    source: str
    reason: str
    match: str


def load_panel_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("test panel config must contain a mapping")
    return config


def normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper().replace("U", "T")
    invalid = sorted(set(normalized) - IUPAC_DNA)
    if invalid:
        raise ValueError(f"unsupported sequence symbols: {''.join(invalid)}")
    if not normalized:
        raise ValueError("empty sequence")
    return normalized


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accession_from_header(header: str) -> str:
    first = header.split()[0] if header else ""
    parts = [part for part in first.split("|") if part]
    for part in reversed(parts):
        if ACCESSION_PATTERN.match(part):
            return part
    return first


def accession_root(accession: str) -> str:
    return accession.upper().split(".", 1)[0]


def species_from_header(header: str, accession: str) -> str:
    words = header.split()
    if words and accession in words[0]:
        words = words[1:]
    organism_words = []
    for word in words:
        clean = word.strip("[](),;:")
        if not organism_words:
            if re.fullmatch(r"[A-Z][A-Za-z-]+", clean):
                organism_words.append(clean)
        elif re.fullmatch(r"[a-z][A-Za-z.-]+", clean):
            organism_words.append(clean)
            break
        else:
            break
    return " ".join(organism_words)


def genus_from_header(header: str, accession: str) -> str:
    species = species_from_header(header, accession)
    return species.split()[0] if species else ""


def stream_fasta(path: str | Path, source_type: str = "external-fasta") -> Iterable[PanelRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    open_fn = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    header = None
    chunks: list[str] = []
    with open_fn(path, mode) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield make_record(header, "".join(chunks), str(path), source_type)
                header = line[1:].strip()
                chunks = []
            elif header is None:
                raise ValueError(f"sequence found before FASTA header in {path}")
            else:
                chunks.append(line)
    if header is not None:
        yield make_record(header, "".join(chunks), str(path), source_type)


def make_record(header: str, sequence: str, source: str, source_type: str) -> PanelRecord:
    accession = accession_from_header(header)
    if not accession:
        raise ValueError(f"missing accession in FASTA header: {header}")
    return PanelRecord(accession, header, normalize_sequence(sequence), source, source_type)


def build_training_index(training_fasta: str | Path) -> TrainingIndex:
    accessions: dict[str, str] = {}
    sequence_hashes: dict[str, str] = {}
    genera: set[str] = set()
    records = 0
    for record in stream_fasta(training_fasta, source_type="training-fasta"):
        root = accession_root(record.accession)
        accessions.setdefault(root, record.accession)
        sequence_hashes.setdefault(sequence_sha256(record.sequence), record.accession)
        genus = genus_from_header(record.header, record.accession)
        if genus:
            genera.add(genus.lower())
        records += 1
    if not records:
        raise ValueError(f"training FASTA contains no records: {training_fasta}")
    return TrainingIndex(accessions, sequence_hashes, genera, records)


def download_accession(
    accession: str,
    download_dir: str | Path,
    email: str,
    overwrite: bool = False,
) -> Path:
    if not email:
        raise ValueError("an NCBI email is required when downloading accessions")
    target = Path(download_dir) / f"{accession}.fasta"
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    Entrez.email = email
    with Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text") as handle:
        text = handle.read()
    if not text.lstrip().startswith(">"):
        raise ValueError(f"NCBI did not return FASTA for {accession}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(target)
    return target


def collect_candidates(
    accessions: list[str],
    external_fastas: list[str],
    download_dir: str | Path,
    email: str,
    overwrite_downloads: bool = False,
) -> list[PanelRecord]:
    records: list[PanelRecord] = []
    for accession in accessions:
        path = download_accession(accession, download_dir, email, overwrite_downloads)
        downloaded = list(stream_fasta(path, source_type="ncbi-accession"))
        if len(downloaded) != 1:
            raise ValueError(f"expected one FASTA record for {accession}, found {len(downloaded)}")
        records.append(
            PanelRecord(
                downloaded[0].accession,
                downloaded[0].header,
                downloaded[0].sequence,
                f"NCBI nuccore:{accession}",
                "ncbi-accession",
            )
        )
    for fasta in external_fastas:
        records.extend(stream_fasta(fasta))
    if not records:
        raise ValueError("provide at least one accession or external FASTA")
    return records


def evaluate_candidates(
    candidates: Iterable[PanelRecord],
    training: TrainingIndex,
    development_accessions: Iterable[str] = (),
) -> tuple[list[dict[str, str | int]], list[Rejection]]:
    accepted: list[dict[str, str | int]] = []
    rejected: list[Rejection] = []
    panel_accessions: dict[str, str] = {}
    panel_hashes: dict[str, str] = {}
    development_roots = {accession_root(item) for item in development_accessions}

    for record in candidates:
        root = accession_root(record.accession)
        forward_hash = sequence_sha256(record.sequence)
        reverse_hash = sequence_sha256(reverse_complement(record.sequence))
        reason = ""
        match = ""
        if root in development_roots:
            reason = "development-accession"
            match = record.accession
        elif root in training.accessions:
            reason = "accession-present-in-training"
            match = training.accessions[root]
        elif forward_hash in training.sequence_hashes:
            reason = "exact-sequence-present-in-training"
            match = training.sequence_hashes[forward_hash]
        elif reverse_hash in training.sequence_hashes:
            reason = "reverse-complement-present-in-training"
            match = training.sequence_hashes[reverse_hash]
        elif root in panel_accessions:
            reason = "duplicate-panel-accession"
            match = panel_accessions[root]
        elif forward_hash in panel_hashes or reverse_hash in panel_hashes:
            reason = "duplicate-panel-sequence"
            match = panel_hashes.get(forward_hash, panel_hashes.get(reverse_hash, ""))

        if reason:
            rejected.append(Rejection(record.accession, record.source, reason, match))
            continue

        species = species_from_header(record.header, record.accession)
        genus = genus_from_header(record.header, record.accession)
        labels = ["unseen-accession"]
        if genus and genus.lower() not in training.genera:
            labels.append("unseen-genus")
        accepted.append(
            {
                "accession": record.accession,
                "species": species,
                "genus": genus,
                "length": len(record.sequence),
                "source": record.source,
                "source_type": record.source_type,
                "sequence_sha256": forward_hash,
                "reverse_complement_sha256": reverse_hash,
                "labels": ";".join(labels),
            }
        )
        panel_accessions[root] = record.accession
        panel_hashes[forward_hash] = record.accession
        panel_hashes[reverse_hash] = record.accession
    return accepted, rejected


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_test_panel(
    training_fasta: str,
    candidates: list[PanelRecord],
    manifest_path: str,
    rejection_path: str,
    metadata_path: str,
    development_accessions: Iterable[str] = (),
) -> tuple[list[dict[str, str | int]], list[Rejection]]:
    training = build_training_index(training_fasta)
    development_accessions = list(development_accessions)
    accepted, rejected = evaluate_candidates(candidates, training, development_accessions)
    write_csv(
        manifest_path,
        accepted,
        [
            "accession",
            "species",
            "genus",
            "length",
            "source",
            "source_type",
            "sequence_sha256",
            "reverse_complement_sha256",
            "labels",
        ],
    )
    write_csv(
        rejection_path,
        [asdict(row) for row in rejected],
        ["accession", "source", "reason", "match"],
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_fasta": training_fasta,
        "training_fasta_sha256": file_sha256(training_fasta),
        "training_records_scanned": training.records,
        "training_genera": len(training.genera),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "development_accessions": development_accessions,
        "manifest": manifest_path,
        "rejections": rejection_path,
    }
    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return accepted, rejected


def configured_email(config: dict, cli_email: str | None) -> str:
    return cli_email or config.get("entrez_email", "") or os.environ.get("NCBI_EMAIL", "")
