from dataclasses import replace

import torch

from src.dna.data import DnaTokenizer
from src.dna.training import (
    PRESETS,
    build_recovery_batch,
    recovery_probability_for_epoch,
)


def test_recovery_probability_schedule():
    preset = PRESETS["quick"]

    assert abs(recovery_probability_for_epoch(0, preset) - 0.02) < 1e-9
    assert abs(recovery_probability_for_epoch(1, preset) - 0.06) < 1e-9
    assert abs(recovery_probability_for_epoch(2, preset) - 0.10) < 1e-9
    assert abs(recovery_probability_for_epoch(5, preset) - 0.10) < 1e-9
    assert recovery_probability_for_epoch(0, PRESETS["quick-control"]) == 0.0

    immediate = replace(preset, recovery_warmup_epochs=0)
    assert recovery_probability_for_epoch(0, immediate) == 0.10


def test_recovery_corruption_uses_previous_logit():
    torch.manual_seed(0)
    tokenizer = DnaTokenizer()
    a_id = tokenizer.vocab["A"]
    c_id = tokenizer.vocab["C"]
    g_id = tokenizer.vocab["G"]
    t_id = tokenizer.vocab["T"]

    input_ids = torch.tensor([[a_id, c_id, g_id, t_id, a_id]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 1, 1, 0]])
    logits = torch.zeros(1, 5, tokenizer.vocab_size)

    logits[0, 2, a_id] = 8.0
    logits[0, 3, c_id] = 8.0

    corrupted, recovery_mask, changed_count, eligible_count = build_recovery_batch(
        input_ids,
        attention_mask,
        loss_mask,
        logits,
        tokenizer,
        corruption_probability=1.0,
    )

    assert eligible_count == 2
    assert changed_count == 2
    assert corrupted[0, 3].item() == a_id
    assert corrupted[0, 4].item() == c_id
    assert recovery_mask.tolist() == [[False, False, False, True, False]]


def test_recovery_ignores_prompt_and_special_tokens():
    torch.manual_seed(0)
    tokenizer = DnaTokenizer()
    a_id = tokenizer.vocab["A"]
    c_id = tokenizer.vocab["C"]
    eos_id = tokenizer.eos_token_id

    input_ids = torch.tensor([[a_id, c_id, eos_id, a_id]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 1, 1, 0]])
    logits = torch.zeros(1, 4, tokenizer.vocab_size)
    logits[..., a_id] = 8.0

    corrupted, _, changed_count, eligible_count = build_recovery_batch(
        input_ids,
        attention_mask,
        loss_mask,
        logits,
        tokenizer,
        corruption_probability=1.0,
    )

    assert eligible_count == 1
    assert changed_count == 0
    assert corrupted.tolist() == input_ids.tolist()


if __name__ == "__main__":
    test_recovery_probability_schedule()
    test_recovery_corruption_uses_previous_logit()
    test_recovery_ignores_prompt_and_special_tokens()
    print("Recovery training tests passed.")
