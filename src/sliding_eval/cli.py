import argparse

from src.sliding_eval.fasta import PlastidRecord, find_genomes, find_record_by_accession
from src.sliding_eval.generation import generate_windows
from src.sliding_eval.regions import RegionMap, infer_regions
from src.sliding_eval.windows import build_windows, write_windows_csv


DEFAULT_FASTA = "data/plastid/refseq_full/refseq_plastids_all.fna.gz"
DEFAULT_ACCESSION = "NC_053550.1"
EXPECTED_LENGTH = 157396
DEFAULT_OUTPUT_DIR = "outputs/sliding_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window evaluation on plastid FASTA records.")
    parser.add_argument("--fasta_file", type=str, default=DEFAULT_FASTA)
    parser.add_argument("--accession", type=str, default=DEFAULT_ACCESSION)
    parser.add_argument("--genus", type=str, default="Rosa")
    parser.add_argument("--max_genomes", type=int, default=None)
    parser.add_argument("--list_genomes", action="store_true")
    parser.add_argument("--check_regions", action="store_true")
    parser.add_argument("--prompt_length", type=int, default=512)
    parser.add_argument("--generate_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max_windows_per_genome", type=int, default=None)
    parser.add_argument("--window_starts", type=str, default=None)
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--decoding_mode",
        choices=["raw_greedy", "constrained_greedy", "sampled"],
        default="raw_greedy",
    )
    parser.add_argument("--sample", action="store_true", help="Alias for --decoding_mode sampled.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def parse_window_starts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    starts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        starts.append(int(item))
    return starts


def resolve_decoding(args: argparse.Namespace) -> tuple[str, bool, float, int | None]:
    mode = "sampled" if args.sample else args.decoding_mode
    requested_ngram = args.no_repeat_ngram_size
    normalized_ngram = requested_ngram if requested_ngram is not None and requested_ngram >= 2 else None

    if mode == "raw_greedy":
        if args.repetition_penalty not in (None, 1.0) or normalized_ngram is not None:
            raise ValueError("raw_greedy does not allow repetition penalties or n-gram blocking")
        return mode, False, 1.0, None

    if mode == "constrained_greedy":
        repetition_penalty = 1.25 if args.repetition_penalty is None else args.repetition_penalty
        no_repeat_ngram_size = 8 if requested_ngram is None else normalized_ngram
        return mode, False, repetition_penalty, no_repeat_ngram_size

    repetition_penalty = 1.0 if args.repetition_penalty is None else args.repetition_penalty
    return mode, True, repetition_penalty, normalized_ngram


def print_genome_list(genus: str, total_matches: int, records: list[PlastidRecord]) -> None:
    print(f"Genus: {genus}")
    print(f"Matching genomes: {total_matches}")
    print(f"Listed genomes: {len(records)}")
    print()
    for idx, record in enumerate(records, start=1):
        print(f"{idx}\t{record.accession}\t{record.length}\t{record.header}")


def print_region_map(record: PlastidRecord, region_map: RegionMap) -> None:
    print("Region check")
    print(f"  Accession: {record.accession}")
    print(f"  Length: {record.length}")
    print(f"  Status: {region_map.status}")
    for region in sorted(region_map.regions, key=lambda item: item.start):
        start = region.start
        end = region.end - 1
        display_start = start % record.length
        display_end = end % record.length
        print(f"  {region.name}: {display_start}-{display_end} length={region.length}")


def print_selected_record(record: PlastidRecord) -> None:
    print("Selected plastid genome")
    print(f"  Accession: {record.accession}")
    print(f"  Length: {record.length}")
    print(f"  Header: {record.header}")


def validate_default_record(record: PlastidRecord) -> None:
    if record.accession == DEFAULT_ACCESSION and record.length != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected {DEFAULT_ACCESSION} to be {EXPECTED_LENGTH} bp, found {record.length} bp"
        )


def main() -> None:
    args = parse_args()
    decoding_mode, do_sample, repetition_penalty, no_repeat_ngram_size = resolve_decoding(args)

    if args.list_genomes:
        total_matches, records = find_genomes(args.fasta_file, args.genus, args.max_genomes)
        print_genome_list(args.genus, total_matches, records)
        return

    record = find_record_by_accession(args.fasta_file, args.accession)
    print_selected_record(record)
    validate_default_record(record)

    if args.check_regions:
        region_map = infer_regions(record.sequence)
        print()
        print_region_map(record, region_map)

    if args.dry_run or args.checkpoint:
        region_map = infer_regions(record.sequence)
        window_starts = parse_window_starts(args.window_starts)
        windows = build_windows(
            record,
            region_map,
            args.prompt_length,
            args.generate_length,
            args.stride,
            args.max_windows_per_genome,
            window_starts,
            circular=args.circular,
        )
        if args.checkpoint:
            generate_windows(
                windows,
                checkpoint=args.checkpoint,
                generate_length=args.generate_length,
                batch_size=args.batch_size,
                do_sample=do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                decoding_mode=decoding_mode,
                seed=args.seed,
            )
        output_path = write_windows_csv(record, windows, args.output_dir)
        print()
        print("Sliding-window evaluation" if args.checkpoint else "Sliding-window dry run")
        print(f"  Windows: {len(windows)}")
        print(f"  Prompt length: {args.prompt_length}")
        print(f"  Target length: {args.generate_length}")
        print(f"  Stride: {args.stride}")
        print(f"  Circular: {'yes' if args.circular else 'no'}")
        if window_starts is not None:
            print(f"  Window starts: {','.join(str(start) for start in window_starts)}")
        if args.checkpoint:
            print(f"  Checkpoint: {args.checkpoint}")
            print(f"  Decoding mode: {decoding_mode}")
            print(f"  Repetition penalty: {repetition_penalty}")
            print(f"  No-repeat n-gram size: {no_repeat_ngram_size or 0}")
        print(f"  CSV: {output_path}")
