import csv
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

from src.biological_eval.config import require_keys
from src.biological_eval.sliding import load_panel_records, selected_panel
from src.dna.checkpoint import load_model
from src.dna.generation import generate_bases
from src.sliding_eval.windows import slice_sequence


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one value is required")
    return values


def target_starts(genome_length: int, prompt_length: int, stride: int, max_targets: int | None) -> list[int]:
    starts = []
    for window_start in range(0, genome_length, stride):
        starts.append((window_start + prompt_length) % genome_length)
        if max_targets is not None and len(starts) >= max_targets:
            break
    return starts


def exact_identity(generated: str, truth: str) -> float:
    if not generated or not truth:
        return 0.0
    matches = sum(a == b for a, b in zip(generated, truth))
    return 100.0 * matches / min(len(generated), len(truth))


def gc_fraction(sequence: str) -> float:
    return sum(base in {"G", "C"} for base in sequence) / len(sequence) if sequence else 0.0


def longest_run(sequence: str) -> int:
    best = 0
    previous = None
    current = 0
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        best = max(best, current)
    return best


def kmer_diversity(sequence: str, k: int = 8) -> float:
    if len(sequence) < k:
        return 0.0
    total = len(sequence) - k + 1
    return len({sequence[index : index + k] for index in range(total)}) / total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_context(rows: list[dict[str, Any]], reports_dir: str) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["accession"], int(row["context_length"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (accession, context_length), group_rows in sorted(groups.items()):
        accuracies = [float(row["accuracy_percent"]) for row in group_rows]
        gc_diffs = [float(row["gc_difference_percent"]) for row in group_rows]
        longest_runs = [int(row["longest_run"]) for row in group_rows]
        diversities = [float(row["kmer_diversity"]) for row in group_rows]
        summary.append(
            {
                "accession": accession,
                "context_length": context_length,
                "rows": len(group_rows),
                "avg_accuracy_percent": f"{mean(accuracies):.2f}",
                "min_accuracy_percent": f"{min(accuracies):.2f}",
                "max_accuracy_percent": f"{max(accuracies):.2f}",
                "avg_gc_difference_percent": f"{mean(gc_diffs):.2f}",
                "max_longest_run": max(longest_runs),
                "avg_kmer_diversity": f"{mean(diversities):.4f}",
            }
        )

    overall: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        overall.setdefault(int(row["context_length"]), []).append(row)
    for context_length, group_rows in sorted(overall.items()):
        accuracies = [float(row["accuracy_percent"]) for row in group_rows]
        gc_diffs = [float(row["gc_difference_percent"]) for row in group_rows]
        longest_runs = [int(row["longest_run"]) for row in group_rows]
        diversities = [float(row["kmer_diversity"]) for row in group_rows]
        summary.append(
            {
                "accession": "all",
                "context_length": context_length,
                "rows": len(group_rows),
                "avg_accuracy_percent": f"{mean(accuracies):.2f}",
                "min_accuracy_percent": f"{min(accuracies):.2f}",
                "max_accuracy_percent": f"{max(accuracies):.2f}",
                "avg_gc_difference_percent": f"{mean(gc_diffs):.2f}",
                "max_longest_run": max(longest_runs),
                "avg_kmer_diversity": f"{mean(diversities):.4f}",
            }
        )

    write_csv(Path(reports_dir) / "context_length_summary.csv", summary)


@torch.no_grad()
def run_context(
    config: dict[str, Any],
    output_dir: str,
    reports_dir: str,
    max_genomes: int | None,
    max_targets: int | None,
    seeds_value: str | None,
    context_lengths_value: str | None,
) -> None:
    require_keys(config, ["raw_fasta", "checkpoint", "panel", "context_lengths", "generation_length"])
    seeds = parse_int_list(seeds_value, [int(config["primary_decoding"]["seed"])])
    context_lengths = parse_int_list(context_lengths_value, [int(value) for value in config["context_lengths"]])
    panel = selected_panel(config, max_genomes)
    records = load_panel_records(config["raw_fasta"], panel)
    model, tokenizer, device = load_model(config["checkpoint"])
    rows: list[dict[str, Any]] = []

    tasks = []
    for item in panel:
        record = records.get(item["accession"])
        if record is None:
            continue
        starts = target_starts(
            record.length,
            int(config["prompt_length"]),
            int(config["stride"]),
            max_targets,
        )
        for target_start in starts:
            truth = slice_sequence(record.sequence, target_start, int(config["generation_length"]), circular=True)
            for context_length in context_lengths:
                prompt_start = target_start - context_length
                prompt = slice_sequence(record.sequence, prompt_start, context_length, circular=True)
                for seed in seeds:
                    tasks.append((record, target_start, context_length, prompt, truth, seed))

    for record, target_start, context_length, prompt, truth, seed in tqdm(tasks, desc="Context generations"):
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        output = generate_bases(
            model,
            tokenizer,
            prompt_ids,
            max_new_bases=int(config["generation_length"]),
            sampling_temperature=float(config["primary_decoding"]["temperature"]),
        )
        generated = tokenizer.decode(output[0, len(prompt) :].tolist(), stop_at_eos=False)
        rows.append(
            {
                "accession": record.accession,
                "target_start": target_start,
                "context_length": context_length,
                "seed": seed,
                "accuracy_percent": f"{exact_identity(generated, truth):.2f}",
                "gc_difference_percent": f"{100.0 * abs(gc_fraction(generated) - gc_fraction(truth)):.2f}",
                "longest_run": longest_run(generated),
                "kmer_diversity": f"{kmer_diversity(generated):.4f}",
                "prompt_length": len(prompt),
                "generated_length": len(generated),
                "true_suffix": truth,
                "generated_suffix": generated,
            }
        )

    output_path = Path(output_dir) / "context_eval.csv"
    write_csv(output_path, rows)
    summarize_context(rows, reports_dir)
    print("Context stage complete")
    print(f"  Rows: {len(rows)}")
    print(f"  Output: {output_path}")
    print(f"  Summary: {Path(reports_dir) / 'context_length_summary.csv'}")


def base_frequency_baseline(sequence: str) -> str:
    counts = Counter(sequence)
    return counts.most_common(1)[0][0] if counts else "A"


@torch.no_grad()
def run_topk(
    config: dict[str, Any],
    output_dir: str,
    reports_dir: str,
    max_genomes: int | None,
    max_targets: int | None,
) -> None:
    require_keys(config, ["raw_fasta", "checkpoint", "panel", "prompt_length", "generation_length", "stride"])
    panel = selected_panel(config, max_genomes)
    records = load_panel_records(config["raw_fasta"], panel)
    model, tokenizer, device = load_model(config["checkpoint"])
    if getattr(model, "prediction_unit", "base") != "base":
        raise ValueError("top-k base evaluation currently requires a one-base checkpoint")
    base_ids = torch.tensor([tokenizer.vocab[base] for base in "ACGT"], device=device)
    rows: list[dict[str, Any]] = []

    for item in panel:
        record = records.get(item["accession"])
        if record is None:
            continue
        starts = target_starts(record.length, int(config["prompt_length"]), int(config["stride"]), max_targets)
        baseline_base = base_frequency_baseline(record.sequence)
        baseline_id = tokenizer.vocab[baseline_base]
        for target_start in starts:
            context_start = target_start - int(config["prompt_length"])
            sequence = slice_sequence(
                record.sequence,
                context_start,
                int(config["prompt_length"]) + int(config["generation_length"]),
                circular=True,
            )
            input_ids = torch.tensor([tokenizer.encode(sequence[:-1])], dtype=torch.long, device=device)
            targets = torch.tensor([tokenizer.encode(sequence[1:])], dtype=torch.long, device=device)
            logits = model(input_ids, attention_mask=torch.ones_like(input_ids))
            scored_logits = logits[:, int(config["prompt_length"]) - 1 :, :]
            scored_targets = targets[:, int(config["prompt_length"]) - 1 :]
            base_logits = scored_logits.index_select(dim=-1, index=base_ids)
            probabilities = base_logits.softmax(dim=-1)
            ranked = base_logits.argsort(dim=-1, descending=True)
            true_base_index = (scored_targets.unsqueeze(-1) == base_ids).float().argmax(dim=-1)
            top1 = ranked[..., 0]
            top2 = ranked[..., :2]
            true_prob = probabilities.gather(-1, true_base_index.unsqueeze(-1)).squeeze(-1)
            total = scored_targets.numel()
            rows.append(
                {
                    "accession": record.accession,
                    "target_start": target_start,
                    "bases": total,
                    "top1_accuracy_percent": f"{100.0 * (top1 == true_base_index).sum().item() / total:.2f}",
                    "top2_accuracy_percent": f"{100.0 * (top2 == true_base_index.unsqueeze(-1)).any(dim=-1).sum().item() / total:.2f}",
                    "mean_true_base_probability": f"{true_prob.mean().item():.4f}",
                    "base_frequency_baseline_percent": f"{100.0 * (scored_targets == baseline_id).sum().item() / total:.2f}",
                    "baseline_base": baseline_base,
                }
            )

    output_path = Path(reports_dir) / "topk_summary.csv"
    write_csv(output_path, rows)
    print("Top-k stage complete")
    print(f"  Rows: {len(rows)}")
    print(f"  Output: {output_path}")
