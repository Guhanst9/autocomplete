import argparse

from src.dna.training import TRANSFORMER_PRESETS, run_training


DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz"
DEFAULT_HOLDOUT = "NC_053550.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal Transformer DNA model.")
    parser.add_argument("--preset", choices=sorted(TRANSFORMER_PRESETS), required=True)
    parser.add_argument("--fasta-file", default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
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
        model_type="transformer",
        prediction_unit=args.prediction_unit,
    )


if __name__ == "__main__":
    main()
