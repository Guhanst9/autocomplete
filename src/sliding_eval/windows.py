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


def build_windows(
    record: PlastidRecord,
    region_map: RegionMap,
    prompt_length: int,
    generate_length: int,
    stride: int,
    max_windows: Optional[int] = None,
) -> list[SlidingWindow]:
    if prompt_length <= 0 or generate_length <= 0 or stride <= 0:
        raise ValueError("prompt_length, generate_length, and stride must be positive")

    total_length = prompt_length + generate_length
    windows: list[SlidingWindow] = []
    for start in range(0, record.length - total_length + 1, stride):
        prompt_start = start
        prompt_end = start + prompt_length - 1
        target_start = start + prompt_length
        target_end = start + total_length - 1
        windows.append(
            SlidingWindow(
                window_start=start,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
                target_start=target_start,
                target_end=target_end,
                region=label_interval(target_start, target_end + 1, region_map, record.length),
                prompt=record.sequence[prompt_start : prompt_end + 1],
                true_suffix=record.sequence[target_start : target_end + 1],
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


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
                    "prompt": window.prompt,
                    "generated_suffix": window.generated_suffix,
                    "true_suffix": window.true_suffix,
                }
            )
    return output_path
