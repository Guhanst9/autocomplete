import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class PlastidRecord:
    accession: str
    header: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def normalize_dna(sequence: str) -> str:
    sequence = sequence.upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in sequence)


def parse_accession(header: str) -> str:
    return header.split()[0] if header else ""


def stream_fasta(path: str) -> Iterable[PlastidRecord]:
    open_fn = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    header: Optional[str] = None
    chunks: list[str] = []

    with open_fn(path, mode) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = normalize_dna("".join(chunks))
                    yield PlastidRecord(parse_accession(header), header, sequence)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

        if header is not None:
            sequence = normalize_dna("".join(chunks))
            yield PlastidRecord(parse_accession(header), header, sequence)


def genus_pattern(genus: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(genus)}\b", re.IGNORECASE)


def find_genomes(
    fasta_file: str,
    genus: str,
    max_genomes: Optional[int] = None,
) -> tuple[int, list[PlastidRecord]]:
    if not Path(fasta_file).exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    pattern = genus_pattern(genus)
    total_matches = 0
    selected: list[PlastidRecord] = []

    for record in stream_fasta(fasta_file):
        if not pattern.search(record.header):
            continue
        total_matches += 1
        if max_genomes is None or len(selected) < max_genomes:
            selected.append(record)

    return total_matches, selected


def find_record_by_accession(fasta_file: str, accession: str) -> PlastidRecord:
    if not Path(fasta_file).exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    for record in stream_fasta(fasta_file):
        if record.accession == accession:
            return record
    raise ValueError(f"Accession not found in FASTA: {accession}")
