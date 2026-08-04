import argparse

from src.biological_eval.config import load_config
from src.biological_eval.baseline import run_baseline_check
from src.biological_eval.prepare import run_prepare
from src.biological_eval.sliding import run_sliding
from src.biological_eval.summary import run_summarize


DEFAULT_CONFIG = "configs/plastid_evaluation.yaml"
DEFAULT_OUTPUT_DIR = "outputs/plastid_biological_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible plastid S4D biological evaluation stages.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("prepare", "baseline-check", "sliding", "summarize"),
        required=True,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", default="reports/plastid_eval")
    parser.add_argument("--max-genomes", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        run_prepare(config, args.output_dir)
    elif args.stage == "baseline-check":
        run_baseline_check(config, args.output_dir)
    elif args.stage == "sliding":
        run_sliding(
            config,
            args.output_dir,
            max_genomes=args.max_genomes,
            max_windows=args.max_windows,
            seeds_value=args.seeds,
            overwrite=args.overwrite,
        )
    elif args.stage == "summarize":
        run_summarize(args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
