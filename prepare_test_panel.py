import argparse

from src.evaluation.test_panel import (
    collect_candidates,
    configured_email,
    load_panel_config,
    prepare_test_panel,
)


DEFAULT_CONFIG = "configs/untouched_test_panel.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an untouched plastid test panel.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--training-fasta")
    parser.add_argument("--accession", action="append", default=[])
    parser.add_argument("--external-fasta", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--rejections")
    parser.add_argument("--metadata")
    parser.add_argument("--download-dir")
    parser.add_argument("--entrez-email")
    parser.add_argument("--overwrite-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_panel_config(args.config)
    training_fasta = args.training_fasta or config["training_fasta"]
    manifest = args.output or config["manifest"]
    rejections = args.rejections or config["rejections"]
    metadata = args.metadata or config["metadata"]
    download_dir = args.download_dir or config["download_dir"]
    accessions = list(config.get("accessions", [])) + args.accession
    external_fastas = list(config.get("external_fastas", [])) + args.external_fasta

    candidates = collect_candidates(
        accessions,
        external_fastas,
        download_dir,
        configured_email(config, args.entrez_email),
        overwrite_downloads=args.overwrite_downloads,
    )
    accepted, rejected = prepare_test_panel(
        training_fasta,
        candidates,
        manifest,
        rejections,
        metadata,
        config.get("development_accessions", []),
    )
    print("Untouched plastid test panel")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Manifest: {manifest}")
    print(f"  Rejections: {rejections}")
    print(f"  Metadata: {metadata}")


if __name__ == "__main__":
    main()
