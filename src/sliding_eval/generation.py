from itertools import groupby
import torch
from tqdm import tqdm

from src.dna.checkpoint import load_model
from src.sliding_eval.windows import SlidingWindow


def exact_identity_percent(generated: str, expected: str) -> float:
    overlap = min(len(generated), len(expected))
    if overlap == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(generated[:overlap], expected[:overlap]))
    return 100.0 * matches / overlap


def longest_homopolymer_run(sequence: str) -> int:
    return max((sum(1 for _ in group) for _, group in groupby(sequence)), default=0)


def gc_difference_percent(generated: str, expected: str) -> float:
    if not generated or not expected:
        return 0.0
    generated_gc = sum(base in {"G", "C"} for base in generated) / len(generated)
    expected_gc = sum(base in {"G", "C"} for base in expected) / len(expected)
    return 100.0 * abs(generated_gc - expected_gc)


@torch.no_grad()
def generate_windows(
    windows: list[SlidingWindow],
    checkpoint: str,
    generate_length: int,
    batch_size: int,
    seed: int,
) -> None:
    model, tokenizer, device = load_model(checkpoint)
    torch.manual_seed(seed)
    for start in tqdm(range(0, len(windows), batch_size), desc="Generate"):
        batch = windows[start : start + batch_size]
        prompt_ids = [tokenizer.encode(window.prompt) for window in batch]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        forbidden = [
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.eos_token_id,
        ]
        if "N" in tokenizer.vocab:
            forbidden.append(tokenizer.vocab["N"])
        output = model.generate(
            prompt_tensor,
            max_new_tokens=generate_length,
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=False,
            forbidden_token_ids=tuple(forbidden),
            min_new_tokens=generate_length,
        )
        for window, token_ids in zip(batch, output.tolist()):
            decoded = tokenizer.decode(token_ids, stop_at_eos=False)
            window.generated_suffix = decoded[len(window.prompt) : len(window.prompt) + generate_length]
            window.generated_length = len(window.generated_suffix)
            window.accuracy_percent = exact_identity_percent(window.generated_suffix, window.true_suffix)
            window.decoding_mode = "raw_greedy"
            window.longest_generated_run = longest_homopolymer_run(window.generated_suffix)
            window.n_count = window.generated_suffix.count("N")
            window.gc_difference_percent = gc_difference_percent(
                window.generated_suffix,
                window.true_suffix,
            )
