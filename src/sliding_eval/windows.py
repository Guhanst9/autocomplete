import csv
import os
from dataclasses import dataclass
from typing import Optional

from src.sliding_eval.fasta import PlastidRecord
from src.sliding_eval.regions import RegionMap, label_interval


@dataclass
class SlidingWindow:
    window_start: int
    prompt_start: int
    prompt_end: int
    target_start: int
    target_end: int
    region: str
    prompt: str
    true_suffix: str
    generated_suffix: str = ""
    generated_length: int | None = None
    accuracy_percent: float | None = None
    decoding_mode: str = ""
    fallback_count: int | None = None
    longest_generated_run: int | None = None
    n_count: int | None = None
    gc_difference_percent: float | None = None


def build_windows(
    record: PlastidRecord,
    region_map: RegionMap,
    prompt_length: int,
    generate_length: int,
    stride: int,
    max_windows: Optional[int] = None,
    window_starts: Optional[list[int]] = None,
    circular: bool = False,
) -> list[SlidingWindow]:
    if prompt_length <= 0 or generate_length <= 0 or stride <= 0:
        raise ValueError("prompt_length, generate_length, and stride must be positive")

    total_length = prompt_length + generate_length
    windows: list[SlidingWindow] = []
    if window_starts is None:
        starts = default_window_starts(record.length, prompt_length, total_length, stride, circular)
    else:
        starts = window_starts

    for start in starts:
        validate_window_start(start, record.length, total_length, circular)
        prompt_start = start % record.length
        prompt_end = (start + prompt_length - 1) % record.length
        target_start_abs = start + prompt_length
        target_end_abs = start + total_length
        target_start = target_start_abs % record.length
        target_end = (target_end_abs - 1) % record.length
        windows.append(
            SlidingWindow(
                window_start=start % record.length,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
                target_start=target_start,
                target_end=target_end,
                region=label_interval(target_start_abs, target_end_abs, region_map, record.length),
                prompt=slice_sequence(record.sequence, start, prompt_length, circular),
                true_suffix=slice_sequence(record.sequence, target_start_abs, generate_length, circular),
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def default_window_starts(
    record_length: int,
    prompt_length: int,
    total_length: int,
    stride: int,
    circular: bool,
) -> list[int]:
    if not circular:
        return list(range(0, record_length - total_length + 1, stride))

    last_prompt_start = record_length - prompt_length
    starts = list(range(0, last_prompt_start + 1, stride))
    if starts and starts[-1] != last_prompt_start:
        starts.append(last_prompt_start)
    elif not starts and last_prompt_start >= 0:
        starts.append(last_prompt_start)
    return starts


def validate_window_start(start: int, record_length: int, total_length: int, circular: bool) -> None:
    if circular:
        if start < 0 or start >= record_length:
            raise ValueError(f"Window start {start} is outside valid circular range 0-{record_length - 1}")
        return

    max_start = record_length - total_length
    if start < 0 or start > max_start:
        raise ValueError(f"Window start {start} is outside valid range 0-{max_start}")


def slice_sequence(sequence: str, start: int, length: int, circular: bool) -> str:
    if not circular:
        return sequence[start : start + length]

    n = len(sequence)
    start = start % n
    if start + length <= n:
        return sequence[start : start + length]
    repeats = (length // n) + 2
    return (sequence * repeats)[start : start + length]


def write_windows_csv(record: PlastidRecord, windows: list[SlidingWindow], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{record.accession}_windows.csv")
    fieldnames = [
        "accession",
        "header",
        "genome_length",
        "window_start",
        "prompt_start",
        "prompt_end",
        "target_start",
        "target_end",
        "region",
        "generated_length",
        "accuracy_percent",
        "decoding_mode",
        "fallback_count",
        "longest_generated_run",
        "n_count",
        "gc_difference_percent",
        "prompt",
        "generated_suffix",
        "true_suffix",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "accession": record.accession,
                    "header": record.header,
                    "genome_length": record.length,
                    "window_start": window.window_start,
                    "prompt_start": window.prompt_start,
                    "prompt_end": window.prompt_end,
                    "target_start": window.target_start,
                    "target_end": window.target_end,
                    "region": window.region,
                    "generated_length": "" if window.generated_length is None else window.generated_length,
                    "accuracy_percent": (
                        "" if window.accuracy_percent is None else f"{window.accuracy_percent:.2f}"
                    ),
                    "decoding_mode": window.decoding_mode,
                    "fallback_count": "" if window.fallback_count is None else window.fallback_count,
                    "longest_generated_run": (
                        "" if window.longest_generated_run is None else window.longest_generated_run
                    ),
                    "n_count": "" if window.n_count is None else window.n_count,
                    "gc_difference_percent": (
                        ""
                        if window.gc_difference_percent is None
                        else f"{window.gc_difference_percent:.2f}"
                    ),
                    "prompt": window.prompt,
                    "generated_suffix": window.generated_suffix,
                    "true_suffix": window.true_suffix,
                }
            )
    return output_path
