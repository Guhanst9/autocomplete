import torch

from src.dna.data import DnaTokenizer as PlastidTokenizer
from src.dna.training import homopolymer_end_loss


def test_penalty_targets_the_end_of_a_long_run():
    tokenizer = PlastidTokenizer()
    a_id = tokenizer.vocab["A"]
    c_id = tokenizer.vocab["C"]
    input_ids = torch.tensor([[a_id] * 8 + [c_id]])
    target_ids = torch.tensor([[a_id] * 7 + [c_id, a_id]])
    loss_mask = torch.ones_like(input_ids)

    high_repeat_logits = torch.zeros(1, 9, tokenizer.vocab_size)
    high_repeat_logits[0, 7, a_id] = 5.0
    low_repeat_logits = high_repeat_logits.clone()
    low_repeat_logits[0, 7, a_id] = -5.0
    low_repeat_logits[0, 7, c_id] = 5.0

    high_loss, high_count = homopolymer_end_loss(
        high_repeat_logits,
        input_ids,
        target_ids,
        loss_mask,
        {tokenizer.vocab[base] for base in "ACGT"},
        min_run=8,
    )
    low_loss, low_count = homopolymer_end_loss(
        low_repeat_logits,
        input_ids,
        target_ids,
        loss_mask,
        {tokenizer.vocab[base] for base in "ACGT"},
        min_run=8,
    )

    assert high_count == low_count == 1
    assert high_loss.item() > low_loss.item()


def test_penalty_ignores_short_or_continuing_runs():
    tokenizer = PlastidTokenizer()
    a_id = tokenizer.vocab["A"]
    logits = torch.zeros(1, 8, tokenizer.vocab_size, requires_grad=True)
    input_ids = torch.tensor([[a_id] * 8])
    target_ids = torch.tensor([[a_id] * 8])
    loss_mask = torch.ones_like(input_ids)

    loss, count = homopolymer_end_loss(
        logits,
        input_ids,
        target_ids,
        loss_mask,
        {tokenizer.vocab[base] for base in "ACGT"},
        min_run=8,
    )

    assert count == 0
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


if __name__ == "__main__":
    test_penalty_targets_the_end_of_a_long_run()
    test_penalty_ignores_short_or_continuing_runs()
    print("Plastid homopolymer loss tests passed.")
