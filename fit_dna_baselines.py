import argparse
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

from src.baselines.checkpoint import FORMAT_VERSION, save_baseline_checkpoint
from src.baselines.markov import fit_count_baselines
from src.dna.data import accession_from_header, stream_fasta
from src.dna.prediction import TripletCodec
from src.evaluation.test_panel import file_sha256


DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit count-based DNA baselines.")
    parser.add_argument("--fasta-file", default=DEFAULT_FASTA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-accession", default="NC_053550.1")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def header_fingerprint(headers: list[str]) -> str:
    digest = hashlib.sha256()
    for header in headers:
        digest.update(header.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_training_headers(
    fasta_file: str,
    holdout_accession: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    headers = []
    for index, (header, sequence) in enumerate(stream_fasta(fasta_file), start=1):
        if accession_from_header(header) == holdout_accession:
            continue
        if not sequence or set(sequence.upper()) - set("ACGT"):
            continue
        headers.append(header)
        if index % 1000 == 0:
            print(f"  Scanned {index:,} records...", flush=True)
    if len(headers) < 2:
        raise ValueError("at least two usable records are required")
    random.Random(seed).shuffle(headers)
    validation_size = max(1, int(round(len(headers) * validation_fraction)))
    validation_size = min(validation_size, len(headers) - 1)
    return headers[validation_size:], headers[:validation_size]


def selected_sequences(fasta_file: str, training_headers: set[str]):
    selected = 0
    for header, sequence in stream_fasta(fasta_file):
        if header not in training_headers:
            continue
        selected += 1
        if selected % 1000 == 0:
            print(f"  Counted {selected:,} training records...", flush=True)
        yield sequence.upper()
    if selected != len(training_headers):
        raise RuntimeError(
            f"loaded {selected} training records, expected {len(training_headers)}"
        )


def fit_baseline_checkpoint(
    fasta_file: str,
    holdout_accession: str,
    validation_fraction: float,
    seed: int,
) -> dict:
    train_headers, val_headers = split_training_headers(
        fasta_file,
        holdout_accession,
        validation_fraction,
        seed,
    )
    base_counts, triplet_counts, markov_counts = fit_count_baselines(
        selected_sequences(fasta_file, set(train_headers))
    )
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_unit": "triplet",
        "triplet_vocabulary": TripletCodec().triplets,
        "base_counts": base_counts,
        "triplet_counts": triplet_counts,
        "markov_counts": markov_counts,
        "training": {
            "fasta_file": fasta_file,
            "fasta_sha256": file_sha256(fasta_file),
            "holdout_accession": holdout_accession,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "train_records": len(train_headers),
            "validation_records": len(val_headers),
            "train_record_fingerprint": header_fingerprint(train_headers),
            "validation_record_fingerprint": header_fingerprint(val_headers),
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint = fit_baseline_checkpoint(
        args.fasta_file,
        args.holdout_accession,
        args.validation_fraction,
        args.seed,
    )
    save_baseline_checkpoint(checkpoint, args.output)
    training = checkpoint["training"]
    print("DNA count baselines")
    print(f"  Training records: {training['train_records']:,}")
    print(f"  Validation records excluded: {training['validation_records']:,}")
    print(f"  Holdout excluded: {training['holdout_accession']}")
    print(f"  Output: {Path(args.output)}")


if __name__ == "__main__":
    main()
