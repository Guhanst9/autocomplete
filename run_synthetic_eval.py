import argparse
import csv
import os
import random
from dataclasses import dataclass

from src.sliding_eval.generation import generate_windows, longest_homopolymer_run
from src.sliding_eval.windows import SlidingWindow


DEFAULT_CHECKPOINT = "outputs/plastid_s4d_v2_recovery_full/best_loss.pt"
DEFAULT_OUTPUT_DIR = "outputs/synthetic_eval"


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    description: str
    prompt: str
    expected: str


def repeat_to_length(pattern: str, length: int) -> str:
    repeats = (length + len(pattern) - 1) // len(pattern)
    return (pattern * repeats)[:length]


def seeded_dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def build_cases(length: int = 512, seed: int = 13) -> list[SyntheticCase]:
    motif_64 = seeded_dna(64, seed)
    copy_block = seeded_dna(length, seed + 1)
    patterns = [
        ("constant_a", "all A bases", "A"),
        ("alternating_at", "alternating A and T", "AT"),
        ("alternating_gc", "alternating G and C", "GC"),
        ("cycle_acgt", "repeating ACGT cycle", "ACGT"),
        ("motif_8", "repeating 8-base motif", "ATGCCGTA"),
        ("motif_64", "repeating seeded 64-base motif", motif_64),
    ]
    cases = [
        SyntheticCase(name, description, repeat_to_length(pattern, length), repeat_to_length(pattern, length))
        for name, description, pattern in patterns
    ]
    cases.append(
        SyntheticCase(
            "copy_512",
            "repeat the complete seeded 512-base prompt",
            copy_block,
            copy_block,
        )
    )
    return cases


def make_windows(cases: list[SyntheticCase]) -> list[SlidingWindow]:
    return [
        SlidingWindow(
            window_start=0,
            prompt_start=0,
            prompt_end=len(case.prompt) - 1,
            target_start=len(case.prompt),
            target_end=len(case.prompt) + len(case.expected) - 1,
            region="synthetic",
            prompt=case.prompt,
            true_suffix=case.expected,
        )
        for case in cases
    ]


def kmer_diversity(sequence: str, k: int = 8) -> float:
    count = len(sequence) - k + 1
    if count <= 0:
        return 0.0
    return len({sequence[start : start + k] for start in range(count)}) / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DNA model on controlled synthetic patterns.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(seed=args.seed)
    results = []

    for decoding_mode in ("raw_greedy", "sampled"):
        windows = make_windows(cases)
        generate_windows(
            windows,
            checkpoint=args.checkpoint,
            generate_length=512,
            batch_size=len(windows),
            seed=args.seed,
            decoding_mode=decoding_mode,
            temperature=args.temperature,
        )
        results.extend(zip(cases, windows))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "synthetic_controls.csv")
    fieldnames = [
        "case",
        "description",
        "decoding_mode",
        "temperature",
        "accuracy_percent",
        "generated_longest_run",
        "expected_longest_run",
        "gc_difference_percent",
        "generated_8mer_diversity",
        "expected_8mer_diversity",
        "prompt",
        "generated_suffix",
        "expected_suffix",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case, window in results:
            writer.writerow(
                {
                    "case": case.name,
                    "description": case.description,
                    "decoding_mode": window.decoding_mode,
                    "temperature": args.temperature if window.decoding_mode == "sampled" else "",
                    "accuracy_percent": f"{window.accuracy_percent:.2f}",
                    "generated_longest_run": window.longest_generated_run,
                    "expected_longest_run": longest_homopolymer_run(case.expected),
                    "gc_difference_percent": f"{window.gc_difference_percent:.2f}",
                    "generated_8mer_diversity": f"{kmer_diversity(window.generated_suffix):.4f}",
                    "expected_8mer_diversity": f"{kmer_diversity(case.expected):.4f}",
                    "prompt": case.prompt,
                    "generated_suffix": window.generated_suffix,
                    "expected_suffix": case.expected,
                }
            )

    print("Synthetic evaluation")
    print(f"  Cases: {len(cases)}")
    print("  Prompt/continuation length: 512/512")
    print("  Decoding modes: raw_greedy, sampled")
    print(f"  Sampled temperature: {args.temperature}")
    print(f"  CSV: {output_path}")


if __name__ == "__main__":
    main()
