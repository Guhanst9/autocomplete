import argparse
import csv
import json
import random
import shutil
import statistics
from itertools import groupby
from pathlib import Path

from Bio.Align import PairwiseAligner

from src.sliding_eval.fasta import find_record_by_accession
from src.sliding_eval.generation import generate_windows
from src.sliding_eval.windows import SlidingWindow, slice_sequence, write_windows_csv


EXPECTED_ROWS = 614
PROMPT_512 = 512
PROMPT_1024 = 1024
TARGET_LENGTH = 512
ALIGNMENT_SCORES = {
    "match": 2.0,
    "mismatch": -1.0,
    "gap_open": -2.0,
    "gap_extension": -0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matched 512-vs-1024 Rosa context evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--fasta-file", required=True)
    parser.add_argument("--accession", default="NC_053550.1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{path} contains {len(rows)} rows, expected {EXPECTED_ROWS}")
    return rows


def validate_baseline_row(row: dict[str, str], row_index: int, genome_length: int) -> None:
    lengths = (len(row["prompt"]), len(row["generated_suffix"]), len(row["true_suffix"]))
    if lengths != (PROMPT_512, TARGET_LENGTH, TARGET_LENGTH):
        raise ValueError(f"baseline row {row_index} has prompt/generated/true lengths {lengths}")
    if row["decoding_mode"] != "sampled":
        raise ValueError(f"baseline row {row_index} is not sampled decoding")
    if int(row["genome_length"]) != genome_length:
        raise ValueError(f"baseline row {row_index} has the wrong genome length")
    target_start = int(row["target_start"])
    target_end = int(row["target_end"])
    if target_end != (target_start + TARGET_LENGTH - 1) % genome_length:
        raise ValueError(f"baseline row {row_index} has inconsistent target coordinates")


def build_matched_windows(
    rows: list[dict[str, str]],
    genome_sequence: str,
) -> list[SlidingWindow]:
    genome_length = len(genome_sequence)
    windows = []
    seen_targets = set()
    for row_index, row in enumerate(rows):
        validate_baseline_row(row, row_index, genome_length)
        target_start = int(row["target_start"])
        if target_start in seen_targets:
            raise ValueError(f"duplicate target start at row {row_index}: {target_start}")
        seen_targets.add(target_start)

        prompt_start = (target_start - PROMPT_1024) % genome_length
        prompt = slice_sequence(genome_sequence, target_start - PROMPT_1024, PROMPT_1024, True)
        true_suffix = slice_sequence(genome_sequence, target_start, TARGET_LENGTH, True)
        if prompt[-PROMPT_512:] != row["prompt"]:
            raise ValueError(f"row {row_index} final 512 prompt bases do not match baseline")
        if true_suffix != row["true_suffix"]:
            raise ValueError(f"row {row_index} true suffix does not match baseline")
        if set(prompt + true_suffix) - set("ACGT"):
            raise ValueError(f"row {row_index} prompt or truth contains a non-ACGT base")

        windows.append(
            SlidingWindow(
                window_start=prompt_start,
                prompt_start=prompt_start,
                prompt_end=(target_start - 1) % genome_length,
                target_start=target_start,
                target_end=int(row["target_end"]),
                region=row["region"],
                region_source=row.get("region_source", ""),
                prompt=prompt,
                true_suffix=true_suffix,
            )
        )
    if len(seen_targets) != EXPECTED_ROWS:
        raise ValueError("target coordinates are not unique")
    return windows


def exact_matches(generated: str, truth: str) -> int:
    return sum(a == b for a, b in zip(generated, truth))


def longest_run(sequence: str) -> int:
    return max((sum(1 for _ in values) for _, values in groupby(sequence)), default=0)


def condition_metrics(rows: list[dict[str, str]], expected_prompt_length: int) -> dict:
    row_matches = []
    first_matches = 0
    last_matches = 0
    runs = []
    invalid_generated = 0
    invalid_prompt = 0
    invalid_truth = 0
    for row_index, row in enumerate(rows):
        prompt = row["prompt"]
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        lengths = (len(prompt), len(generated), len(truth))
        expected = (expected_prompt_length, TARGET_LENGTH, TARGET_LENGTH)
        if lengths != expected:
            raise ValueError(f"row {row_index} has prompt/generated/true lengths {lengths}, expected {expected}")
        matches = exact_matches(generated, truth)
        row_matches.append(matches)
        first_matches += exact_matches(generated[:100], truth[:100])
        last_matches += exact_matches(generated[-100:], truth[-100:])
        runs.append(longest_run(generated))
        invalid_generated += sum(base not in "ACGT" for base in generated)
        invalid_prompt += sum(base not in "ACGT" for base in prompt)
        invalid_truth += sum(base not in "ACGT" for base in truth)
    row_accuracy = [100.0 * value / TARGET_LENGTH for value in row_matches]
    return {
        "rows": len(rows),
        "prompt_length": expected_prompt_length,
        "generated_length": TARGET_LENGTH,
        "true_suffix_length": TARGET_LENGTH,
        "mean_exact_accuracy_percent": statistics.fmean(row_accuracy),
        "median_exact_accuracy_percent": statistics.median(row_accuracy),
        "first_100_accuracy_percent": 100.0 * first_matches / (100 * len(rows)),
        "last_100_accuracy_percent": 100.0 * last_matches / (100 * len(rows)),
        "longest_repeated_base_run": max(runs),
        "rows_with_runs_over_20": sum(run > 20 for run in runs),
        "invalid_generated_base_count": invalid_generated,
        "invalid_prompt_base_count": invalid_prompt,
        "invalid_true_base_count": invalid_truth,
        "row_exact_matches": row_matches,
    }


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = ALIGNMENT_SCORES["match"]
    aligner.mismatch_score = ALIGNMENT_SCORES["mismatch"]
    aligner.open_gap_score = ALIGNMENT_SCORES["gap_open"]
    aligner.extend_gap_score = ALIGNMENT_SCORES["gap_extension"]
    return aligner


def one_alignment(generated: str, truth: str, aligner: PairwiseAligner) -> dict:
    alignment = aligner.align(generated, truth)[0]
    counts = alignment.counts()
    columns = counts.identities + counts.mismatches + counts.gaps
    return {
        "identity_percent": 100.0 * counts.identities / columns if columns else 0.0,
        "gap_columns": int(counts.gaps),
        "score": float(alignment.score),
    }


def alignment_metrics(rows: list[dict[str, str]], shuffle_seed: int) -> tuple[dict, list[dict]]:
    aligner = make_aligner()
    rng = random.Random(shuffle_seed)
    per_row = []
    shuffled_per_row = []
    for row in rows:
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        per_row.append(one_alignment(generated, truth, aligner))
        shuffled = list(generated)
        rng.shuffle(shuffled)
        shuffled_per_row.append(one_alignment("".join(shuffled), truth, aligner))

    summary = {
        "mode": "global",
        "scores": ALIGNMENT_SCORES,
        "identity_denominator": "matches + mismatches + gap columns",
        "mean_aligned_identity_percent": statistics.fmean(row["identity_percent"] for row in per_row),
        "average_gap_columns": statistics.fmean(row["gap_columns"] for row in per_row),
        "mean_alignment_score": statistics.fmean(row["score"] for row in per_row),
        "shuffled_control": {
            "method": "shuffle each generated suffix while preserving its base composition",
            "seed": shuffle_seed,
            "mean_aligned_identity_percent": statistics.fmean(
                row["identity_percent"] for row in shuffled_per_row
            ),
            "average_gap_columns": statistics.fmean(row["gap_columns"] for row in shuffled_per_row),
            "mean_alignment_score": statistics.fmean(row["score"] for row in shuffled_per_row),
        },
    }
    paired = [
        {
            **actual,
            "shuffled_identity_percent": shuffled["identity_percent"],
            "shuffled_gap_columns": shuffled["gap_columns"],
        }
        for actual, shuffled in zip(per_row, shuffled_per_row)
    ]
    return summary, paired


def paired_metrics(matches_512: list[int], matches_1024: list[int]) -> dict:
    deltas = [100.0 * (new - old) / TARGET_LENGTH for old, new in zip(matches_512, matches_1024)]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    positive = sorted((delta for delta in deltas if delta > 0), reverse=True)
    top_10_share = 0.0 if not positive else 100.0 * sum(positive[:10]) / sum(positive)
    if statistics.fmean(deltas) > 0 and wins > len(deltas) / 2:
        interpretation = "Longer real context improves a majority of all windows."
    elif statistics.fmean(deltas) > 0:
        interpretation = (
            "Longer real context raises the mean without improving a majority of all windows; "
            "the gain is concentrated in a smaller set of windows."
        )
    elif statistics.fmean(deltas) == 0:
        interpretation = "Longer real context leaves mean exact accuracy unchanged."
    else:
        interpretation = "Longer real context lowers mean exact accuracy."
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_delta_accuracy_points": statistics.fmean(deltas),
        "median_delta_accuracy_points": statistics.median(deltas),
        "mean_winner_gain_points": statistics.fmean(delta for delta in deltas if delta > 0) if wins else 0.0,
        "mean_loser_change_points": statistics.fmean(delta for delta in deltas if delta < 0) if losses else 0.0,
        "windows_gaining_at_least_5_points": sum(delta >= 5.0 for delta in deltas),
        "windows_losing_at_least_5_points": sum(delta <= -5.0 for delta in deltas),
        "top_10_positive_gain_share_percent": top_10_share,
        "interpretation": interpretation,
        "row_delta_accuracy_points": deltas,
    }


def write_paired_csv(
    path: Path,
    baseline: list[dict[str, str]],
    metrics_512: dict,
    metrics_1024: dict,
    alignment_512: list[dict],
    alignment_1024: list[dict],
) -> None:
    fields = [
        "row_index",
        "target_start",
        "exact_accuracy_512_percent",
        "exact_accuracy_1024_percent",
        "delta_accuracy_points",
        "paired_result",
        "aligned_identity_512_percent",
        "aligned_identity_1024_percent",
        "gap_columns_512",
        "gap_columns_1024",
        "shuffled_identity_512_percent",
        "shuffled_identity_1024_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (old, new) in enumerate(zip(metrics_512["row_exact_matches"], metrics_1024["row_exact_matches"])):
            delta = 100.0 * (new - old) / TARGET_LENGTH
            writer.writerow(
                {
                    "row_index": index,
                    "target_start": baseline[index]["target_start"],
                    "exact_accuracy_512_percent": f"{100.0 * old / TARGET_LENGTH:.8f}",
                    "exact_accuracy_1024_percent": f"{100.0 * new / TARGET_LENGTH:.8f}",
                    "delta_accuracy_points": f"{delta:.8f}",
                    "paired_result": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                    "aligned_identity_512_percent": f"{alignment_512[index]['identity_percent']:.8f}",
                    "aligned_identity_1024_percent": f"{alignment_1024[index]['identity_percent']:.8f}",
                    "gap_columns_512": alignment_512[index]["gap_columns"],
                    "gap_columns_1024": alignment_1024[index]["gap_columns"],
                    "shuffled_identity_512_percent": f"{alignment_512[index]['shuffled_identity_percent']:.8f}",
                    "shuffled_identity_1024_percent": f"{alignment_1024[index]['shuffled_identity_percent']:.8f}",
                }
            )


def summary_markdown(summary: dict) -> str:
    old = summary["prompt_512"]
    new = summary["prompt_1024"]
    paired = summary["paired"]
    lines = [
        "# Matched Rosa context comparison",
        "",
        "| Metric | 512-base prompt | 1,024-base prompt |",
        "|---|---:|---:|",
        f"| Mean exact accuracy | {old['mean_exact_accuracy_percent']:.4f}% | {new['mean_exact_accuracy_percent']:.4f}% |",
        f"| Median exact accuracy | {old['median_exact_accuracy_percent']:.4f}% | {new['median_exact_accuracy_percent']:.4f}% |",
        f"| First 100 accuracy | {old['first_100_accuracy_percent']:.4f}% | {new['first_100_accuracy_percent']:.4f}% |",
        f"| Final 100 accuracy | {old['last_100_accuracy_percent']:.4f}% | {new['last_100_accuracy_percent']:.4f}% |",
        f"| Longest repeated-base run | {old['longest_repeated_base_run']} | {new['longest_repeated_base_run']} |",
        f"| Rows with runs over 20 | {old['rows_with_runs_over_20']} | {new['rows_with_runs_over_20']} |",
        f"| Invalid generated bases | {old['invalid_generated_base_count']} | {new['invalid_generated_base_count']} |",
        f"| Mean global aligned identity | {summary['alignment_512']['mean_aligned_identity_percent']:.4f}% | {summary['alignment_1024']['mean_aligned_identity_percent']:.4f}% |",
        f"| Average global-alignment gap columns | {summary['alignment_512']['average_gap_columns']:.4f} | {summary['alignment_1024']['average_gap_columns']:.4f} |",
        f"| Shuffled-control aligned identity | {summary['alignment_512']['shuffled_control']['mean_aligned_identity_percent']:.4f}% | {summary['alignment_1024']['shuffled_control']['mean_aligned_identity_percent']:.4f}% |",
        "",
        f"Paired windows: **{paired['wins']} wins, {paired['ties']} ties, {paired['losses']} losses** for the 1,024-base prompt.",
        "",
        paired["interpretation"],
        "",
        "Global alignment scoring: match +2, mismatch -1, gap open -2, gap extension -0.5. "
        "Aligned identity uses matches divided by matches, mismatches, and gap columns. "
        "The shuffled control independently shuffles each generated suffix with seed 13 while preserving base composition.",
        "",
        "Exact-position accuracy remains the primary metric; aligned identity is supplemental.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline = read_rows(args.baseline_csv)
    record = find_record_by_accession(args.fasta_file, args.accession)
    windows = build_matched_windows(baseline, record.sequence)
    print(f"Validated {len(windows)} matched target coordinates")
    print("All 1,024-base prompts end with the original 512-base prompt")
    print("All true 512-base suffixes are unchanged")
    if args.validate_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.output_dir / "prompt_512"
    generated_dir = args.output_dir / "prompt_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    baseline_copy = baseline_dir / f"{args.accession}_windows.csv"
    shutil.copyfile(args.baseline_csv, baseline_copy)

    generate_windows(
        windows,
        checkpoint=args.checkpoint,
        generate_length=TARGET_LENGTH,
        batch_size=args.batch_size,
        seed=args.seed,
        decoding_mode="sampled",
        temperature=args.temperature,
    )
    generated_path = Path(write_windows_csv(record, windows, str(generated_dir)))
    baseline_rows = read_rows(baseline_copy)
    generated_rows = read_rows(generated_path)
    for index, (old, new) in enumerate(zip(baseline_rows, generated_rows)):
        if old["target_start"] != new["target_start"] or old["true_suffix"] != new["true_suffix"]:
            raise ValueError(f"row {index} target changed after generation")
        if new["prompt"][-PROMPT_512:] != old["prompt"]:
            raise ValueError(f"row {index} prompt overlap changed after generation")

    metrics_512 = condition_metrics(baseline_rows, PROMPT_512)
    metrics_1024 = condition_metrics(generated_rows, PROMPT_1024)
    aligned_512, aligned_rows_512 = alignment_metrics(baseline_rows, args.seed)
    aligned_1024, aligned_rows_1024 = alignment_metrics(generated_rows, args.seed)
    paired = paired_metrics(metrics_512["row_exact_matches"], metrics_1024["row_exact_matches"])
    metrics_512.pop("row_exact_matches")
    metrics_1024.pop("row_exact_matches")

    summary = {
        "accession": args.accession,
        "checkpoint": args.checkpoint,
        "rows": EXPECTED_ROWS,
        "generation": {
            "decoding_mode": "sampled",
            "temperature": args.temperature,
            "seed": args.seed,
            "seeded_once_before_generation": True,
            "batch_size": args.batch_size,
            "window_order": "original baseline CSV order",
        },
        "validation": {
            "all_target_coordinates_unchanged": True,
            "all_prompt_1024_lengths_valid": True,
            "all_final_512_prompt_bases_match": True,
            "all_generated_lengths_valid": True,
            "all_true_suffixes_unchanged": True,
            "all_true_suffix_lengths_valid": True,
        },
        "prompt_512": metrics_512,
        "prompt_1024": metrics_1024,
        "paired": {key: value for key, value in paired.items() if key != "row_delta_accuracy_points"},
        "alignment_512": aligned_512,
        "alignment_1024": aligned_1024,
        "files": {
            "prompt_512_csv": str(baseline_copy),
            "prompt_1024_csv": str(generated_path),
            "paired_csv": str(args.output_dir / "paired_comparison.csv"),
        },
    }
    write_paired_csv(
        args.output_dir / "paired_comparison.csv",
        baseline_rows,
        {"row_exact_matches": [exact_matches(row["generated_suffix"], row["true_suffix"]) for row in baseline_rows]},
        {"row_exact_matches": [exact_matches(row["generated_suffix"], row["true_suffix"]) for row in generated_rows]},
        aligned_rows_512,
        aligned_rows_1024,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "summary.md").write_text(summary_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
