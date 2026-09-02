import argparse
from pathlib import Path

from src.baselines.checkpoint import load_baseline_checkpoint
from src.evaluation.shared import (
    build_record_windows,
    generate_baseline_windows,
    summarize_rows,
    validate_result_csv,
    write_result_csv,
    write_rows,
)
from src.sliding_eval.fasta import find_record_by_accession


METHODS = [
    ("most-common-base", "most-common-base", None),
    ("triplet-frequency", "triplet-frequency", None),
    ("markov-1", "markov", 1),
    ("markov-3", "markov", 3),
    ("markov-6", "markov", 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate count baselines on development data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta-file", required=True)
    parser.add_argument("--accession", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generate-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--development", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development:
        raise ValueError("pass --development; untouched data must use evaluate_models.py")
    checkpoint = load_baseline_checkpoint(args.checkpoint)
    settings = {
        "prompt_length": args.prompt_length,
        "generation_length": args.generate_length,
        "stride": args.stride,
        "circular": args.circular,
    }
    summary_rows = []
    for name, method, order in METHODS:
        rows = []
        model_dir = Path(args.output_dir) / name
        model_config = {"method": method, "order": order}
        for accession in args.accession:
            record = find_record_by_accession(args.fasta_file, accession)
            windows = build_record_windows(record, settings)
            generate_baseline_windows(windows, model_config, checkpoint)
            csv_path = write_result_csv(record, windows, model_dir)
            rows.extend(validate_result_csv(csv_path, windows))
        summary_rows.append({"model": name, **summarize_rows(rows)})
    summary_path = Path(args.output_dir) / "baseline_comparison.csv"
    write_rows(summary_path, summary_rows)
    print("Development baseline evaluation complete")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
