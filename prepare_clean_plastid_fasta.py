import argparse
import csv
import gzip
import random
from typing import Iterable

DEFAULT_INPUT = "data/plastid/refseq_full/refseq_plastids_all.fna.gz"
DEFAULT_OUTPUT = "data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz"
DEFAULT_REPORT = "data/plastid/refseq_full/refseq_plastids_all_clean_report.csv"
VALID_BASES = {"A", "C", "G", "T", "N"}
REPLACEMENT_BASES = "ACGT"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean plastid FASTA records for no-N training.")
    parser.add_argument("--input_fasta", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_fasta", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--report_csv", type=str, default=DEFAULT_REPORT)
    parser.add_argument("--max_n_fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--line_width", type=int, default=80)
    return parser.parse_args()

def open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def stream_fasta(path: str) -> Iterable[tuple[str, str]]:
    with open_text(path, "rt") as handle:
        header = None
        chunks: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)

def normalize_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace("U", "T")
    return "".join(base if base in VALID_BASES else "N" for base in sequence)

def replace_ns(sequence: str, rng: random.Random) -> str:
    if "N" not in sequence:
        return sequence
    parts = sequence.split("N")
    out: list[str] = []
    for idx, part in enumerate(parts):
        out.append(part)
        if idx < len(parts) - 1:
            out.append(rng.choice(REPLACEMENT_BASES))
    return "".join(out)

def write_fasta_record(handle, header: str, sequence: str, line_width: int) -> None:
    handle.write(f">{header}\n")
    for start in range(0, len(sequence), line_width):
        handle.write(sequence[start : start + line_width] + "\n")

def clean_fasta(
    input_fasta: str,
    output_fasta: str,
    report_csv: str,
    max_n_fraction: float,
    seed: int,
    line_width: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    counts = {
        "records_total": 0,
        "records_kept": 0,
        "records_removed": 0,
        "bases_total": 0,
        "bases_kept": 0,
        "n_total": 0,
        "n_replaced": 0,
    }
    fieldnames = [
        "accession",
        "header",
        "sequence_length",
        "n_count_before",
        "n_fraction_before",
        "status",
        "n_replaced",
    ]

    with open_text(output_fasta, "wt") as out_handle, open(report_csv, "w", newline="") as report_handle:
        writer = csv.DictWriter(report_handle, fieldnames=fieldnames)
        writer.writeheader()

        for header, raw_sequence in stream_fasta(input_fasta):
            sequence = normalize_sequence(raw_sequence)
            length = len(sequence)
            n_count = sequence.count("N")
            n_fraction = n_count / length if length else 0.0
            keep = n_fraction <= max_n_fraction
            accession = header.split()[0] if header else ""

            counts["records_total"] += 1
            counts["bases_total"] += length
            counts["n_total"] += n_count

            if keep:
                cleaned = replace_ns(sequence, rng)
                write_fasta_record(out_handle, header, cleaned, line_width)
                counts["records_kept"] += 1
                counts["bases_kept"] += length
                counts["n_replaced"] += n_count
                status = "kept"
                replaced = n_count
            else:
                counts["records_removed"] += 1
                status = "removed"
                replaced = 0

            writer.writerow(
                {
                    "accession": accession,
                    "header": header,
                    "sequence_length": length,
                    "n_count_before": n_count,
                    "n_fraction_before": f"{n_fraction:.10f}",
                    "status": status,
                    "n_replaced": replaced,
                }
            )

    return counts

def main() -> None:
    args = parse_args()
    counts = clean_fasta(
        input_fasta=args.input_fasta,
        output_fasta=args.output_fasta,
        report_csv=args.report_csv,
        max_n_fraction=args.max_n_fraction,
        seed=args.seed,
        line_width=args.line_width,
    )

    print("Cleaned plastid FASTA")
    print(f"  Input: {args.input_fasta}")
    print(f"  Output: {args.output_fasta}")
    print(f"  Report: {args.report_csv}")
    print(f"  Max N fraction: {args.max_n_fraction}")
    print(f"  Seed: {args.seed}")
    print(f"  Records total: {counts['records_total']:,}")
    print(f"  Records kept: {counts['records_kept']:,}")
    print(f"  Records removed: {counts['records_removed']:,}")
    print(f"  Ns before cleaning: {counts['n_total']:,}")
    print(f"  Ns replaced in kept records: {counts['n_replaced']:,}")

if __name__ == "__main__":
    main()
