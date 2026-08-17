import csv
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

from src.biological_eval.config import require_keys
from src.biological_eval.context_topk import exact_identity, parse_int_list, write_csv
from src.biological_eval.sliding import load_panel_records, selected_panel
from src.dna.checkpoint import load_model
from src.dna.generation import generate_bases
from src.sliding_eval.regions import Region, infer_regions, reverse_complement
from src.sliding_eval.windows import slice_sequence


def inferred_ir_regions(sequence: str) -> tuple[Region, Region, str]:
    region_map = infer_regions(sequence)
    regions = {region.name: region for region in region_map.regions}
    if "IRA" not in regions or "IRB" not in regions:
        raise ValueError("could not infer IRA/IRB regions")
    return regions["IRA"], regions["IRB"], "sequence-inferred"


def paired_offsets(ira: Region, irb: Region, target_length: int, max_pairs: int | None) -> list[int]:
    usable = min(ira.length, irb.length) - target_length
    if usable < 0:
        return []
    count = 20 if max_pairs is None else max_pairs
    if count <= 1:
        return [0]
    step = max(1, usable // (count - 1))
    return [min(offset * step, usable) for offset in range(count)]


def generate_suffix(model, tokenizer, device, prompt: str, length: int, seed: int, temperature: float) -> str:
    torch.manual_seed(seed)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = generate_bases(
        model,
        tokenizer,
        prompt_ids,
        max_new_bases=length,
        sampling_temperature=temperature,
    )
    return tokenizer.decode(output[0, len(prompt) :].tolist(), stop_at_eos=False)


def run_ir(
    config: dict[str, Any],
    output_dir: str,
    reports_dir: str,
    max_genomes: int | None,
    max_pairs: int | None,
    seeds_value: str | None,
) -> None:
    require_keys(config, ["raw_fasta", "checkpoint", "panel", "prompt_length", "generation_length"])
    seeds = parse_int_list(seeds_value, [int(config["primary_decoding"]["seed"])])
    temperature = float(config["primary_decoding"]["temperature"])
    panel = selected_panel(config, max_genomes)
    records = load_panel_records(config["raw_fasta"], panel)
    model, tokenizer, device = load_model(config["checkpoint"])
    rows: list[dict[str, Any]] = []
    tasks = []

    for item in panel:
        record = records.get(item["accession"])
        if record is None:
            continue
        ira, irb, boundary_source = inferred_ir_regions(record.sequence)
        target_length = int(config["generation_length"])
        prompt_length = int(config["prompt_length"])
        for offset in paired_offsets(ira, irb, target_length, max_pairs):
            tasks.append((record, ira, irb, boundary_source, offset, target_length, prompt_length))

    for record, ira, irb, boundary_source, offset, target_length, prompt_length in tqdm(
        tasks, desc="IR pairs", unit="pair"
    ):
        ira_target_start = ira.start + offset
        irb_target_start = irb.start + (irb.length - offset - target_length)
        ira_true = slice_sequence(record.sequence, ira_target_start, target_length, True)
        irb_true = slice_sequence(record.sequence, irb_target_start, target_length, True)
        ira_prompt = slice_sequence(
            record.sequence,
            ira_target_start - prompt_length,
            prompt_length,
            True,
        )
        irb_prompt = slice_sequence(
            record.sequence,
            irb_target_start - prompt_length,
            prompt_length,
            True,
        )
        true_ir_identity = exact_identity(ira_true, reverse_complement(irb_true))
        for seed in seeds:
            ira_generated = generate_suffix(
                model,
                tokenizer,
                device,
                ira_prompt,
                target_length,
                seed,
                temperature,
            )
            irb_generated = generate_suffix(
                model,
                tokenizer,
                device,
                irb_prompt,
                target_length,
                seed,
                temperature,
            )
            rows.append(
                {
                    "accession": record.accession,
                    "boundary_source": boundary_source,
                    "offset": offset,
                    "seed": seed,
                    "true_ir_identity_percent": f"{true_ir_identity:.2f}",
                    "generated_ir_identity_percent": f"{exact_identity(ira_generated, reverse_complement(irb_generated)):.2f}",
                    "ira_identity_to_true_percent": f"{exact_identity(ira_generated, ira_true):.2f}",
                    "irb_identity_to_true_percent": f"{exact_identity(irb_generated, irb_true):.2f}",
                }
            )

    detail_path = Path(output_dir) / "ir_eval.csv"
    write_csv(detail_path, rows)
    summary_rows = []
    by_accession: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_accession.setdefault(row["accession"], []).append(row)
    for accession, accession_rows in sorted(by_accession.items()):
        summary_rows.append(
            {
                "accession": accession,
                "rows": len(accession_rows),
                "avg_true_ir_identity_percent": f"{mean(float(row['true_ir_identity_percent']) for row in accession_rows):.2f}",
                "avg_generated_ir_identity_percent": f"{mean(float(row['generated_ir_identity_percent']) for row in accession_rows):.2f}",
                "avg_ira_identity_to_true_percent": f"{mean(float(row['ira_identity_to_true_percent']) for row in accession_rows):.2f}",
                "avg_irb_identity_to_true_percent": f"{mean(float(row['irb_identity_to_true_percent']) for row in accession_rows):.2f}",
            }
        )
    summary_path = Path(reports_dir) / "ir_summary.csv"
    write_csv(summary_path, summary_rows)
    print("IR stage complete")
    print(f"  Rows: {len(rows)}")
    print(f"  Detail: {detail_path}")
    print(f"  Summary: {summary_path}")
