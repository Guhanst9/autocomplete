import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all.fna.gz"
DEFAULT_ACCESSION = "NC_053550.1"
EXPECTED_LENGTH = 157396

@dataclass
class PlastidRecord:
    accession: str
    header: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window evaluation on plastid FASTA records.")
    parser.add_argument("--fasta_file", type=str, default=DEFAULT_FASTA)
    parser.add_argument("--accession", type=str, default=DEFAULT_ACCESSION)
    parser.add_argument("--genus", type=str, default="Rosa")
    parser.add_argument("--max_genomes", type=int, default=None)
    parser.add_argument("--list_genomes", action="store_true")
    return parser.parse_args()


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

def print_genome_list(genus: str, total_matches: int, records: list[PlastidRecord]) -> None:
    print(f"Genus: {genus}")
    print(f"Matching genomes: {total_matches}")
    print(f"Listed genomes: {len(records)}")
    print()
    for idx, record in enumerate(records, start=1):
        print(f"{idx}\t{record.accession}\t{record.length}\t{record.header}")

def main() -> None:
    args = parse_args()

    if args.list_genomes:
        total_matches, records = find_genomes(args.fasta_file, args.genus, args.max_genomes)
        print_genome_list(args.genus, total_matches, records)
        return

    record = find_record_by_accession(args.fasta_file, args.accession)
    print("Selected plastid genome")
    print(f"  Accession: {record.accession}")
    print(f"  Length: {record.length}")
    print(f"  Header: {record.header}")
    if record.accession == DEFAULT_ACCESSION and record.length != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected {DEFAULT_ACCESSION} to be {EXPECTED_LENGTH} bp, found {record.length} bp"
        )

if __name__ == "__main__":
    main()
