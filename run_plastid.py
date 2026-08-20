import argparse

from src.dna.training import PRESETS, run_training


DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz"
DEFAULT_HOLDOUT = "NC_053550.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the active S4D DNA next-base model.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--fasta-file", default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-additional-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--holdout-accession", default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--prediction-unit",
        choices=("base", "triplet"),
        default="base",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        preset_name=args.preset,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        resume=args.resume,
        holdout_accession=args.holdout_accession,
        prediction_unit=args.prediction_unit,
        max_additional_epochs=args.max_additional_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    main()
