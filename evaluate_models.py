import argparse

from src.evaluation.shared import freeze_evaluation, run_shared_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen DNA models on identical windows.")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze:
        lock = freeze_evaluation(args.test_manifest, args.models, args.output_dir)
        print("Evaluation frozen")
        print(f"  Lock: {lock}")
        print("  No test generation was run")
        return
    summary = run_shared_evaluation(args.test_manifest, args.models, args.output_dir)
    print("Shared evaluation complete")
    print(f"  Summary: {summary}")


if __name__ == "__main__":
    main()
