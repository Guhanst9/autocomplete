import tempfile
from pathlib import Path

import torch

from src.dataloaders.protein import ProteinDataset, ProteinTokenizer
from src.models.s4_model import S4ProteinModel


class FixedLogitModel(S4ProteinModel):
    def __init__(self, vocab_size, eos_token_id):
        super().__init__(
            vocab_size=vocab_size,
            d_model=4,
            d_state=4,
            n_layers=0,
            eos_token_id=eos_token_id,
        )

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        # eos is underpredicted so higher eos weight should raise loss
        logits[..., self.eos_token_id] = -4.0
        return logits


class EosFirstModel(S4ProteinModel):
    def __init__(self, vocab_size, eos_token_id, fallback_token_id):
        super().__init__(
            vocab_size=vocab_size,
            d_model=4,
            d_state=4,
            n_layers=0,
            eos_token_id=eos_token_id,
        )
        self.fallback_token_id = fallback_token_id

    def forward(self, input_ids, attention_mask=None):
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., self.fallback_token_id] = 1.0
        logits[..., self.eos_token_id] = 5.0
        return logits


def test_eos_loss_weight_increases_bad_eos_loss():
    tokenizer = ProteinTokenizer()
    model = FixedLogitModel(tokenizer.vocab_size, tokenizer.eos_token_id)

    input_ids = torch.tensor([[tokenizer.vocab["A"], tokenizer.vocab["C"]]])
    target_ids = torch.tensor([[tokenizer.vocab["C"], tokenizer.eos_token_id]])
    attention_mask = torch.tensor([[1, 1]])
    loss_mask = torch.tensor([[1, 1]])

    base = model.compute_loss(
        input_ids,
        target_ids,
        attention_mask,
        loss_mask=loss_mask,
        objective="autocomplete",
        eos_loss_weight=1.0,
    )
    weighted = model.compute_loss(
        input_ids,
        target_ids,
        attention_mask,
        loss_mask=loss_mask,
        objective="autocomplete",
        eos_loss_weight=8.0,
    )

    assert weighted.item() > base.item()
    print(f"EOS weighting works: base={base.item():.4f}, weighted={weighted.item():.4f}")


def test_end_prefix_sampling_moves_loss_start_near_end():
    tokenizer = ProteinTokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "tiny.fasta"
        fasta.write_text(">seq\nACDEFGHIKL\n")
        dataset = ProteinDataset(
            str(fasta),
            tokenizer=tokenizer,
            l_max=16,
            objective="autocomplete",
            end_prefix_prob=1.0,
            end_prefix_min_fraction=0.80,
            end_prefix_max_fraction=1.00,
        )

        _, _, _, loss_mask = dataset[0]
        first_scored_pos = int(torch.nonzero(loss_mask, as_tuple=False)[0].item())
        sampled_prefix_len = first_scored_pos + 1

    assert sampled_prefix_len >= 8
    print(f"End-prefix sampling works: sampled_prefix_len={sampled_prefix_len}")


def test_min_new_tokens_blocks_early_eos():
    tokenizer = ProteinTokenizer()
    model = EosFirstModel(
        tokenizer.vocab_size,
        tokenizer.eos_token_id,
        tokenizer.vocab["A"],
    )
    prompt = torch.tensor([[tokenizer.vocab["M"]]])

    out = model.generate(
        prompt,
        max_new_tokens=3,
        do_sample=False,
        use_recurrent=False,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=True,
        min_new_tokens=2,
    )[0].tolist()
    generated = out[1:]

    assert generated[:2] == [tokenizer.vocab["A"], tokenizer.vocab["A"]]
    assert generated[2] == tokenizer.eos_token_id
    print("min_new_tokens blocks early EOS")


def test_generation_reports_constraint_fallback():
    model = S4ProteinModel(
        vocab_size=5,
        d_model=4,
        d_state=4,
        n_layers=0,
        eos_token_id=2,
        max_length=8,
    )
    prompt = torch.tensor([[4, 3, 4, 4]])

    output, diagnostics = model.generate(
        prompt,
        max_new_tokens=1,
        do_sample=False,
        use_recurrent=False,
        eos_token_id=2,
        stop_at_eos=False,
        forbidden_token_ids=(0, 1, 2),
        no_repeat_ngram_size=2,
        return_diagnostics=True,
    )

    generated_token = output[0, -1].item()
    assert generated_token in {3, 4}
    assert diagnostics.fallback_counts == (1,)
    print("constraint fallback is reported and permanent bans are preserved")


def test_generation_default_return_stays_tensor():
    model = S4ProteinModel(
        vocab_size=5,
        d_model=4,
        d_state=4,
        n_layers=0,
        eos_token_id=2,
        max_length=8,
    )
    prompt = torch.tensor([[3]])
    output = model.generate(
        prompt,
        max_new_tokens=1,
        do_sample=False,
        use_recurrent=False,
        stop_at_eos=False,
    )

    assert isinstance(output, torch.Tensor)
    print("generation default return remains a tensor")


if __name__ == "__main__":
    test_eos_loss_weight_increases_bad_eos_loss()
    test_end_prefix_sampling_moves_loss_start_near_end()
    test_min_new_tokens_blocks_early_eos()
    test_generation_reports_constraint_fallback()
    test_generation_default_return_stays_tensor()
    print("All protein objective tests passed!")
