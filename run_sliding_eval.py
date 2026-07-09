import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all.fna.gz"
DEFAULT_ACCESSION = "NC_053550.1"
EXPECTED_LENGTH = 157396
DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

@dataclass
class PlastidRecord:
    accession: str
    header: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

@dataclass
class Region:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

@dataclass
class RegionMap:
    regions: list[Region]
    status: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window evaluation on plastid FASTA records.")
    parser.add_argument("--fasta_file", type=str, default=DEFAULT_FASTA)
    parser.add_argument("--accession", type=str, default=DEFAULT_ACCESSION)
    parser.add_argument("--genus", type=str, default="Rosa")
    parser.add_argument("--max_genomes", type=int, default=None)
    parser.add_argument("--list_genomes", action="store_true")
    parser.add_argument("--check_regions", action="store_true")
    return parser.parse_args()


def normalize_dna(sequence: str) -> str:
    sequence = sequence.upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in sequence)

def parse_accession(header: str) -> str:
    return header.split()[0] if header else ""

def reverse_complement(sequence: str) -> str:
    return sequence.translate(DNA_COMPLEMENT)[::-1]

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

def extend_reverse_repeat(sequence: str, left_start: int, right_seed_start: int, seed_length: int) -> Region:
    n = len(sequence)
    left_extra = 0
    while (
        left_start - 1 - left_extra >= 0
        and right_seed_start + seed_length + left_extra < n
        and sequence[left_start - 1 - left_extra]
        == sequence[right_seed_start + seed_length + left_extra].translate(DNA_COMPLEMENT)
    ):
        left_extra += 1

    right_extra = 0
    while (
        left_start + seed_length + right_extra < n
        and right_seed_start - 1 - right_extra >= 0
        and sequence[left_start + seed_length + right_extra]
        == sequence[right_seed_start - 1 - right_extra].translate(DNA_COMPLEMENT)
    ):
        right_extra += 1

    start = left_start - left_extra
    end = left_start + seed_length + right_extra
    return Region("repeat", start, end)

def infer_regions(
    sequence: str,
    seed_length: int = 80,
    scan_step: int = 50,
    min_ir_length: int = 10000,
    min_ir_spacing: int = 10000,
) -> RegionMap:
    positions: dict[str, list[int]] = {}
    n = len(sequence)
    for start in range(0, n - seed_length + 1):
        seed = sequence[start : start + seed_length]
        if "N" in seed:
            continue
        positions.setdefault(seed, []).append(start)

    best_pair: tuple[int, int, int, int] | None = None
    best_length = 0
    for left_start in range(0, n - seed_length + 1, scan_step):
        seed = sequence[left_start : left_start + seed_length]
        if "N" in seed:
            continue
        rc_seed = reverse_complement(seed)
        for right_start in positions.get(rc_seed, []):
            if abs(left_start - right_start) < min_ir_spacing:
                continue
            left = extend_reverse_repeat(sequence, left_start, right_start, seed_length)
            right = extend_reverse_repeat(sequence, right_start, left_start, seed_length)
            if left.length < min_ir_length or right.length < min_ir_length:
                continue
            first, second = sorted([left, right], key=lambda region: region.start)
            if first.length > best_length:
                best_pair = (first.start, first.end, second.start, second.end)
                best_length = first.length

    if best_pair is None:
        return RegionMap([Region("unknown", 0, n)], "unknown")

    ir_a_start, ir_a_end, ir_b_start, ir_b_end = best_pair
    gap_between = Region("single_copy", ir_a_end, ir_b_start)
    gap_wrap = Region("single_copy", ir_b_end, n + ir_a_start)
    lsc_gap, ssc_gap = sorted([gap_between, gap_wrap], key=lambda region: region.length, reverse=True)

    regions = [
        Region("IRA", ir_a_start, ir_a_end),
        Region("IRB", ir_b_start, ir_b_end),
        Region("LSC", lsc_gap.start, lsc_gap.end),
        Region("SSC", ssc_gap.start, ssc_gap.end),
    ]
    return RegionMap(regions, "inferred")

def print_region_map(record: PlastidRecord, region_map: RegionMap) -> None:
    print("Region check")
    print(f"  Accession: {record.accession}")
    print(f"  Length: {record.length}")
    print(f"  Status: {region_map.status}")
    for region in sorted(region_map.regions, key=lambda item: item.start):
        start = region.start
        end = region.end - 1
        display_start = start % record.length
        display_end = end % record.length
        print(
            f"  {region.name}: {display_start}-{display_end} "
            f"length={region.length}"
        )

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
    if args.check_regions:
        region_map = infer_regions(record.sequence)
        print()
        print_region_map(record, region_map)

if __name__ == "__main__":
    main()
