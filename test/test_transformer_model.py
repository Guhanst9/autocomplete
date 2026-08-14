import _path  # noqa: F401

import io
import os
from unittest.mock import patch

import torch

from src.dna.checkpoint import build_model_from_config
from src.dna.data import DnaTokenizer, DnaWindowDataset
from src.dna.training import TRANSFORMER_PRESETS, build_sequence_model, parameter_count
from src.models.s4_model import S4SequenceModel
from src.models.transformer_model import TransformerSequenceModel


def tiny_model() -> TransformerSequenceModel:
    torch.manual_seed(13)
    return TransformerSequenceModel(
        vocab_size=7,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        dropout=0.0,
        pad_token_id=0,
        mask_token_id=1,
        eos_token_id=2,
        max_length=16,
    )


def test_output_shape_and_causality():
    model = tiny_model().eval()
    first = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]])
    second = first.clone()
    second[:, 4:] = torch.tensor([[6, 5, 4, 3]])

    first_logits = model(first)
    second_logits = model(second)

    assert first_logits.shape == (1, 8, 7)
    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6)
    assert not torch.allclose(first_logits[:, 4:], second_logits[:, 4:])


def test_cached_and_uncached_generation_match():
    model = tiny_model().eval()
    prompt = torch.tensor([[3, 4, 5, 6, 3]])
    forbidden = (0, 1, 2)

    cached = model.generate(
        prompt,
        max_new_tokens=6,
        forbidden_token_ids=forbidden,
        stop_at_eos=False,
        use_recurrent=True,
    )
    uncached = model.generate(
        prompt,
        max_new_tokens=6,
        forbidden_token_ids=forbidden,
        stop_at_eos=False,
        use_recurrent=False,
    )

    assert torch.equal(cached, uncached)
    assert cached.shape == (1, 11)
    assert set(cached[0, 5:].tolist()).issubset({3, 4, 5, 6})


def test_sampled_generation_is_seeded():
    model = tiny_model().eval()
    prompt = torch.tensor([[3, 4, 5, 6]])
    options = {
        "max_new_tokens": 8,
        "forbidden_token_ids": (0, 1, 2),
        "stop_at_eos": False,
        "sampling_temperature": 0.8,
    }

    torch.manual_seed(13)
    first = model.generate(prompt, **options)
    torch.manual_seed(13)
    second = model.generate(prompt, **options)

    assert torch.equal(first, second)
    assert set(first[0, 4:].tolist()).issubset({3, 4, 5, 6})


def test_context_and_next_token_alignment():
    tokenizer = DnaTokenizer()
    dataset = DnaWindowDataset(
        fasta_file="synthetic",
        tokenizer=tokenizer,
        l_max=8,
        stride=None,
        max_windows=1,
        windows_per_record=1,
        prefix_min_fraction=0.25,
        prefix_max_fraction=0.70,
        seed=13,
        records=[("synthetic", "ACGTACGT")],
    )

    with patch("src.dna.data.random.randint", return_value=4):
        input_ids, target_ids, attention_mask, loss_mask = dataset[0]

    assert input_ids[:8].tolist() == tokenizer.encode("ACGTACGT")
    assert target_ids[:7].tolist() == tokenizer.encode("CGTACGT")
    assert attention_mask.tolist() == [1] * 8
    assert loss_mask.tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert target_ids[3].item() == tokenizer.vocab["A"]


def test_full_parameter_count():
    tokenizer = DnaTokenizer()
    with torch.device("meta"):
        model = build_sequence_model(
            tokenizer,
            TRANSFORMER_PRESETS["full"],
            model_type="transformer",
        )
    assert parameter_count(model) == 16_416_576


def test_checkpoint_reload_and_legacy_default():
    model = tiny_model().eval()
    config = {
        "model_type": "transformer",
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_length": 16,
    }
    buffer = io.BytesIO()
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu")
    restored = build_model_from_config(checkpoint["model_config"], DnaTokenizer())
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert isinstance(restored, TransformerSequenceModel)

    legacy = build_model_from_config(
        {
            "d_model": 8,
            "d_state": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "kernel_type": "diag",
            "model_variant": "s4d_v2",
            "l_max": 8,
        },
        DnaTokenizer(),
    )
    assert isinstance(legacy, S4SequenceModel)


def test_tiny_mps_training_step():
    if os.environ.get("RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available():
        return
    model = tiny_model().to("mps").train()
    input_ids = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]], device="mps")
    target_ids = torch.tensor([[4, 5, 6, 3, 4, 5, 6, 0]], device="mps")
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0]], device="mps")
    logits = model(input_ids, attention_mask=attention_mask)
    loss = model.compute_loss(
        input_ids,
        target_ids,
        attention_mask,
        loss_mask=loss_mask,
        logits=logits,
    )
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


if __name__ == "__main__":
    test_output_shape_and_causality()
    test_cached_and_uncached_generation_match()
    test_sampled_generation_is_seeded()
    test_context_and_next_token_alignment()
    test_full_parameter_count()
    test_checkpoint_reload_and_legacy_default()
    test_tiny_mps_training_step()
    print("Transformer model tests passed.")
