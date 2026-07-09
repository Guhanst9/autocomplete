from typing import Optional

import torch
from tqdm import tqdm

from run_plastid import PlastidTokenizer
from src.models.s4_model import S4ProteinModel
from src.sliding_eval.windows import SlidingWindow


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str) -> tuple[S4ProteinModel, PlastidTokenizer, torch.device]:
    device = get_device()
    tokenizer = PlastidTokenizer()
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("model_config", {})
    model = S4ProteinModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 448),
        d_state=config.get("d_state", 64),
        n_layers=config.get("n_layers", 10),
        kernel_type=config.get("kernel_type", "diag"),
        bidirectional=False,
        l_max=config.get("l_max"),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=config.get("l_max", 1024),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, device


def exact_identity_percent(generated: str, expected: str) -> float:
    overlap = min(len(generated), len(expected))
    if overlap == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(generated[:overlap], expected[:overlap]))
    return 100.0 * matches / overlap


@torch.no_grad()
def generate_windows(
    windows: list[SlidingWindow],
    checkpoint: str,
    generate_length: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    repetition_penalty: float,
    no_repeat_ngram_size: Optional[int],
) -> None:
    model, tokenizer, device = load_model(checkpoint)
    for start in tqdm(range(0, len(windows), batch_size), desc="Generate"):
        batch = windows[start : start + batch_size]
        prompt_ids = [tokenizer.encode(window.prompt) for window in batch]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        output = model.generate(
            prompt_tensor,
            max_new_tokens=generate_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=False,
            forbidden_token_ids=(
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
                tokenizer.eos_token_id,
                tokenizer.vocab["N"],
            ),
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=generate_length,
        )
        for window, token_ids in zip(batch, output.tolist()):
            decoded = tokenizer.decode(token_ids, stop_at_eos=False)
            window.generated_suffix = decoded[len(window.prompt) : len(window.prompt) + generate_length]
            window.generated_length = len(window.generated_suffix)
            window.accuracy_percent = exact_identity_percent(window.generated_suffix, window.true_suffix)
