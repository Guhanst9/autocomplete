import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from Bio.Seq import Seq

from src.biological_eval.annotations import FeatureRecord, interval_overlap
from src.biological_eval.summary import generated_manifest_rows, read_csv, write_csv
from src.sliding_eval.regions import reverse_complement


TARGET_GENES = {"rbcL", "psbA", "matK", "rpoB", "trnH-GUG", "rrn16", "rrn23"}


def load_features(path: str) -> dict[str, list[FeatureRecord]]:
    features: dict[str, list[FeatureRecord]] = defaultdict(list)
    for row in read_csv(Path(path)):
        feature = FeatureRecord(
            accession=row["accession"],
            feature_type=row["feature_type"],
            gene=row["gene"],
            strand=int(row["strand"]),
            start=int(row["start"]),
            end=int(row["end"]),
        )
        features[feature.accession].append(feature)
    return features


def feature_label(feature: FeatureRecord) -> str:
    if feature.gene in TARGET_GENES:
        return feature.gene
    if feature.feature_type == "CDS":
        return "all_CDS"
    if feature.feature_type == "tRNA":
        return "all_tRNA"
    if feature.feature_type == "rRNA":
        return "all_rRNA"
    return feature.feature_type


def slice_circular(sequence: str, start: int, length: int) -> str:
    if not sequence or length <= 0:
        return ""
    n = len(sequence)
    start %= n
    if start + length <= n:
        return sequence[start : start + length]
    repeats = (length // n) + 2
    return (sequence * repeats)[start : start + length]


def overlapping_feature_segment(
    window: dict[str, str],
    feature: FeatureRecord,
    genome_length: int,
) -> tuple[int, int, int] | None:
    target_start = int(window["target_start"])
    target_end = (int(window["target_end"]) + 1) % genome_length
    overlap = interval_overlap(target_start, target_end, feature.start, feature.end, genome_length)
    if overlap <= 0:
        return None

    target_abs_start = int(window["window_start"]) + len(window["prompt"])
    target_abs_end = target_abs_start + len(window["true_suffix"])
    feature_abs_start = feature.start
    feature_abs_end = feature.end
    if feature_abs_start < target_abs_start % genome_length and target_abs_start >= genome_length:
        feature_abs_start += genome_length
        feature_abs_end += genome_length

    start_abs = max(target_abs_start, feature_abs_start)
    end_abs = min(target_abs_end, feature_abs_end)
    if end_abs <= start_abs:
        return None
    suffix_offset = start_abs - target_abs_start
    feature_offset = start_abs - feature_abs_start
    return suffix_offset, end_abs - start_abs, feature_offset


def translated_identity(generated: str, truth: str) -> tuple[float | None, int | None]:
    usable = min(len(generated), len(truth))
    usable -= usable % 3
    if usable < 3:
        return None, None
    generated_aa = str(Seq(generated[:usable]).translate(to_stop=False))
    truth_aa = str(Seq(truth[:usable]).translate(to_stop=False))
    matches = sum(a == b for a, b in zip(generated_aa, truth_aa))
    identity = 100.0 * matches / len(truth_aa) if truth_aa else None
    internal_stops = generated_aa[:-1].count("*") if generated_aa else 0
    return identity, internal_stops


def cds_metrics(window: dict[str, str], feature: FeatureRecord, genome_length: int) -> tuple[float | None, int | None]:
    segment = overlapping_feature_segment(window, feature, genome_length)
    if segment is None:
        return None, None
    suffix_offset, length, feature_offset = segment
    if feature_offset % 3 != 0:
        return None, None
    generated = window["generated_suffix"][suffix_offset : suffix_offset + length]
    truth = window["true_suffix"][suffix_offset : suffix_offset + length]
    if feature.strand == -1:
        generated = reverse_complement(generated)
        truth = reverse_complement(truth)
    return translated_identity(generated, truth)


def load_window_rows(output_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in generated_manifest_rows(output_dir):
        for row in read_csv(Path(manifest_row["csv_path"])):
            merged = dict(row)
            merged["group"] = manifest_row["group"]
            merged["exposure"] = manifest_row["exposure"]
            merged["seed"] = manifest_row["seed"]
            rows.append(merged)
    return rows


def run_features(output_dir: str, reports_dir: str) -> None:
    feature_path = Path(output_dir) / "annotation_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Run annotations first; missing {feature_path}")
    features = load_features(str(feature_path))
    windows = load_window_rows(output_dir)
    if not windows:
        raise ValueError("No sliding windows found. Run sliding first.")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        accession = window["accession"]
        genome_length = int(window["genome_length"])
        target_start = int(window["target_start"])
        target_end = (int(window["target_end"]) + 1) % genome_length
        for feature in features.get(accession, []):
            overlap = interval_overlap(target_start, target_end, feature.start, feature.end, genome_length)
            if overlap <= 0:
                continue
            labels = {feature_label(feature)}
            if feature.gene in TARGET_GENES:
                labels.add(feature.gene)
            for label in labels:
                aa_identity, internal_stops = (None, None)
                if feature.feature_type == "CDS":
                    aa_identity, internal_stops = cds_metrics(window, feature, genome_length)
                item = dict(window)
                item["feature_label"] = label
                item["feature_type"] = feature.feature_type
                item["gene"] = feature.gene
                item["feature_overlap_bases"] = overlap
                item["aa_identity_percent"] = aa_identity
                item["internal_stop_codons"] = internal_stops
                grouped[(accession, label, feature.feature_type)].append(item)

    summary_rows: list[dict[str, Any]] = []
    for (accession, label, feature_type), rows in sorted(grouped.items()):
        accuracies = [float(row["accuracy_percent"]) for row in rows]
        aa_values = [row["aa_identity_percent"] for row in rows if row["aa_identity_percent"] is not None]
        stop_values = [row["internal_stop_codons"] for row in rows if row["internal_stop_codons"] is not None]
        summary_rows.append(
            {
                "accession": accession,
                "feature_label": label,
                "feature_type": feature_type,
                "rows": len(rows),
                "avg_accuracy_percent": f"{mean(accuracies):.2f}",
                "avg_aa_identity_percent": "" if not aa_values else f"{mean(aa_values):.2f}",
                "internal_stop_codons": "" if not stop_values else sum(stop_values),
                "total_overlap_bases": sum(int(row["feature_overlap_bases"]) for row in rows),
            }
        )

    output_path = Path(reports_dir) / "feature_summary.csv"
    write_csv(
        output_path,
        [
            "accession",
            "feature_label",
            "feature_type",
            "rows",
            "avg_accuracy_percent",
            "avg_aa_identity_percent",
            "internal_stop_codons",
            "total_overlap_bases",
        ],
        summary_rows,
    )
    print("Feature stage complete")
    print(f"  Windows: {len(windows)}")
    print(f"  Feature groups: {len(summary_rows)}")
    print(f"  Summary: {output_path}")
