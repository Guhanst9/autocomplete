import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from Bio.Align import PairwiseAligner

from src.baselines.frequency import most_common_index
from src.dna.checkpoint import load_model
from src.dna.prediction import TripletCodec
from src.sliding_eval.windows import SlidingWindow


ALIGNMENT_SCORES = {
    "match": 2.0,
    "mismatch": -1.0,
    "gap_open": -2.0,
    "gap_extension": -0.5,
}


def make_global_aligner(scores: dict[str, float] | None = None) -> PairwiseAligner:
    scores = scores or ALIGNMENT_SCORES
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = scores["match"]
    aligner.mismatch_score = scores["mismatch"]
    aligner.open_gap_score = scores["gap_open"]
    aligner.extend_gap_score = scores["gap_extension"]
    return aligner


def alignment_identity(generated: str, truth: str, aligner: PairwiseAligner) -> float:
    counts = aligner.align(generated, truth)[0].counts()
    columns = counts.identities + counts.mismatches + counts.gaps
    return 100.0 * counts.identities / columns if columns else 0.0


def add_alignment_identities(
    rows: list[dict[str, str]],
    scores: dict[str, float] | None = None,
) -> None:
    aligner = make_global_aligner(scores)
    for row in rows:
        row["alignment_identity_percent"] = str(
            alignment_identity(row["generated_suffix"], row["true_suffix"], aligner)
        )


def recursive_triplet_accuracy(rows: Iterable[dict[str, str]]) -> tuple[int, int]:
    correct = total = 0
    for row in rows:
        generated = row["generated_suffix"]
        truth = row["true_suffix"]
        for start in range(0, min(len(generated), len(truth)) - 2, 3):
            correct += generated[start : start + 3] == truth[start : start + 3]
            total += 1
    return correct, total


def distance_bin_rows(
    rows: list[dict[str, str]],
    group_keys: list[str],
    width: int = 50,
) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        length = len(group[0]["true_suffix"])
        for start in range(0, length, width):
            end = min(start + width, length)
            matches = bases = 0
            for row in group:
                generated = row["generated_suffix"][start:end]
                truth = row["true_suffix"][start:end]
                matches += sum(left == right for left, right in zip(generated, truth))
                bases += len(truth)
            output.append(
                {
                    **dict(zip(group_keys, key)),
                    "distance_start": start,
                    "distance_end": end - 1,
                    "windows": len(group),
                    "bases": bases,
                    "accuracy_percent": 100.0 * matches / bases,
                }
            )
    return output


def _teacher_sequences(windows: list[SlidingWindow]) -> list[str]:
    sequences = [window.prompt + window.true_suffix for window in windows]
    if any(set(sequence) - set("ACGT") for sequence in sequences):
        raise ValueError("teacher-forced windows must contain only A/C/G/T")
    return sequences


@torch.no_grad()
def neural_teacher_metrics(
    checkpoint: str,
    windows: list[SlidingWindow],
    batch_size: int,
    model_bundle: tuple | None = None,
) -> dict:
    model, tokenizer, device = model_bundle or load_model(checkpoint)
    if model.prediction_unit != "triplet":
        raise ValueError("shared teacher-forced evaluation requires a triplet checkpoint")
    codec = TripletCodec(model.output_tokens)
    sequences = _teacher_sequences(windows)
    total_loss = 0.0
    correct = total = 0
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch_sequences = sequences[start : start + batch_size]
        inputs = torch.tensor(
            [tokenizer.encode(sequence) for sequence in batch_sequences],
            dtype=torch.long,
            device=device,
        )
        logits = model(inputs, attention_mask=torch.ones_like(inputs))
        prompt_length = len(batch_windows[0].prompt)
        target_length = len(batch_windows[0].true_suffix)
        scored_logits = logits[:, prompt_length - 1 : prompt_length + target_length - 3, :]
        targets = torch.tensor(
            [
                [codec.encode(window.true_suffix[offset : offset + 3]) for offset in range(target_length - 2)]
                for window in batch_windows
            ],
            dtype=torch.long,
            device=device,
        )
        total_loss += F.cross_entropy(
            scored_logits.flatten(0, 1),
            targets.flatten(),
            reduction="sum",
        ).item()
        correct += (scored_logits.argmax(dim=-1) == targets).sum().item()
        total += targets.numel()
    average_loss = total_loss / total
    return {
        "teacher_forced_triplets": total,
        "teacher_forced_exact_triplet_accuracy_percent": 100.0 * correct / total,
        "teacher_forced_triplet_cross_entropy": average_loss,
        "base_normalized_perplexity": math.exp(average_loss / 3.0),
        "probability_method": "model-softmax",
    }


def _smoothed_probability(counts: list[int], target: int, alpha: float) -> float:
    return (counts[target] + alpha) / (sum(counts) + alpha * len(counts))


def _markov_row(history: str, order: int, checkpoint: dict) -> list[int]:
    for length in range(min(order, len(history)), 0, -1):
        row = checkpoint["markov_counts"].get(str(length), {}).get(history[-length:])
        if row and sum(row):
            return row
    return checkpoint["triplet_counts"]


def baseline_teacher_metrics(
    model_config: dict,
    checkpoint: dict,
    windows: list[SlidingWindow],
    smoothing: float,
) -> dict:
    if smoothing <= 0:
        raise ValueError("baseline smoothing must be positive")
    codec = TripletCodec(checkpoint["triplet_vocabulary"])
    method = model_config["method"]
    base_counts = checkpoint["base_counts"]
    total_loss = 0.0
    correct = total = 0
    for window in windows:
        history = window.prompt
        for offset in range(len(window.true_suffix) - 2):
            truth = window.true_suffix[offset : offset + 3]
            target = codec.encode(truth)
            if method == "most-common-base":
                predicted = "ACGT"[most_common_index(base_counts)] * 3
                probability = math.prod(
                    _smoothed_probability(base_counts, "ACGT".index(base), smoothing)
                    for base in truth
                )
            else:
                counts = checkpoint["triplet_counts"]
                if method == "markov":
                    counts = _markov_row(history, int(model_config["order"]), checkpoint)
                predicted = codec.decode(max(range(len(counts)), key=counts.__getitem__))
                probability = _smoothed_probability(counts, target, smoothing)
            correct += predicted == truth
            total += 1
            total_loss -= math.log(probability)
            history += window.true_suffix[offset]
    average_loss = total_loss / total
    return {
        "teacher_forced_triplets": total,
        "teacher_forced_exact_triplet_accuracy_percent": 100.0 * correct / total,
        "teacher_forced_triplet_cross_entropy": average_loss,
        "base_normalized_perplexity": math.exp(average_loss / 3.0),
        "probability_method": f"add-{smoothing:g}-smoothed-counts",
    }
