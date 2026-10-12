import argparse

from src.biological_eval.annotations import run_annotations
from src.biological_eval.config import load_config
from src.biological_eval.features import run_features
from src.biological_eval.context_topk import run_context, run_topk
from src.biological_eval.ir import run_ir
from src.biological_eval.reporting import run_figures, run_report
from src.biological_eval.baseline import run_baseline_check
from src.biological_eval.prepare import run_prepare
from src.biological_eval.sliding import run_relabel, run_sliding
from src.biological_eval.summary import run_summarize


DEFAULT_CONFIG = "configs/plastid_evaluation.yaml"
DEFAULT_OUTPUT_DIR = "outputs/s4d_base_16.57m_recovery/evaluations/biological_panel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible plastid S4D biological evaluation stages.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "baseline-check",
            "annotations",
            "sliding",
            "relabel",
            "summarize",
            "features",
            "context",
            "topk",
            "ir",
            "figures",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", default="reports/plastid_eval")
    parser.add_argument("--max-genomes", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--context-lengths", default=None)
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
    elif args.stage == "annotations":
        run_annotations(
            config,
            args.output_dir,
            max_genomes=args.max_genomes,
            overwrite=args.overwrite,
        )
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
    elif args.stage == "relabel":
        run_relabel(config, args.output_dir, max_genomes=args.max_genomes)
    elif args.stage == "features":
        run_features(args.output_dir, args.reports_dir)
    elif args.stage == "context":
        run_context(
            config,
            args.output_dir,
            args.reports_dir,
            max_genomes=args.max_genomes,
            max_targets=args.max_windows,
            seeds_value=args.seeds,
            context_lengths_value=args.context_lengths,
        )
    elif args.stage == "topk":
        run_topk(
            config,
            args.output_dir,
            args.reports_dir,
            max_genomes=args.max_genomes,
            max_targets=args.max_windows,
        )
    elif args.stage == "ir":
        run_ir(
            config,
            args.output_dir,
            args.reports_dir,
            max_genomes=args.max_genomes,
            max_pairs=args.max_windows,
            seeds_value=args.seeds,
        )
    elif args.stage == "figures":
        run_figures(args.reports_dir)
    elif args.stage == "report":
        run_report(config, args.reports_dir)


if __name__ == "__main__":
    main()
