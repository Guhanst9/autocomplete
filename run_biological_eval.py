import argparse

from src.biological_eval.config import load_config
from src.biological_eval.baseline import run_baseline_check
from src.biological_eval.prepare import run_prepare


DEFAULT_CONFIG = "configs/plastid_evaluation.yaml"
DEFAULT_OUTPUT_DIR = "outputs/plastid_biological_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible plastid S4D biological evaluation stages.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("prepare", "baseline-check"), required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        run_prepare(config, args.output_dir)
    elif args.stage == "baseline-check":
        run_baseline_check(config, args.output_dir)


if __name__ == "__main__":
    main()
