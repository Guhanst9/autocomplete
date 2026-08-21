from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dna.data import DnaTokenizer
from src.dna.prediction import TripletCodec
from src.dna.training import (
    PRESETS,
    build_recursive_recovery_batch,
    build_optimizer,
    build_sequence_model,
    train_one_epoch,
)


class RecordingTripletModel(torch.nn.Module):
    def __init__(self, output_tokens):
        super().__init__()
        self.prediction_unit = "triplet"
        self.output_tokens = output_tokens
        self.prefixes = []

    def generate_triplets(
        self,
        prompt_ids,
        triplet_base_ids,
        max_new_bases,
        sampling_temperature=None,
    ):
        self.prefixes.append(prompt_ids.detach().clone())
        next_base = prompt_ids[:, -1:]
        generated = next_base.repeat(1, max_new_bases)
        return torch.cat((prompt_ids, generated), dim=1)


def recovery_inputs(tokenizer):
    sequence = tokenizer.encode("ACGT" * 8)
    input_ids = torch.tensor([sequence, sequence], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[:, 8:] = 1
    return input_ids, attention_mask, loss_mask


def test_recursive_block_is_causal_detached_and_reproducible():
    tokenizer = DnaTokenizer(include_n=False)
    codec = TripletCodec()
    model = RecordingTripletModel(codec.triplets)
    input_ids, attention_mask, loss_mask = recovery_inputs(tokenizer)

    torch.manual_seed(13)
    first = build_recursive_recovery_batch(
        model,
        input_ids,
        attention_mask,
        loss_mask,
        tokenizer,
        block_length=6,
    )
    first_corrupted, first_mask, first_changed, _ = first
    prefix_length = model.prefixes[-1].size(1)

    modified_future = input_ids.clone()
    modified_future[:, prefix_length:] = tokenizer.vocab["T"]
    torch.manual_seed(13)
    second = build_recursive_recovery_batch(
        model,
        modified_future,
        attention_mask,
        loss_mask,
        tokenizer,
        block_length=6,
    )
    second_corrupted, _, _, _ = second

    assert torch.equal(model.prefixes[-2], model.prefixes[-1])
    assert torch.equal(
        first_corrupted[:, prefix_length : prefix_length + 6],
        second_corrupted[:, prefix_length : prefix_length + 6],
    )
    assert first_corrupted.grad_fn is None
    assert first_changed > 0
    first_changed_position = torch.nonzero(first_corrupted[0] != input_ids[0])[0].item()
    assert not first_mask[0, :first_changed_position].any()
    assert first_mask[0, first_changed_position:].all()
    assert model.training


def test_recursive_presets_preserve_model_shape():
    tokenizer = DnaTokenizer(include_n=False)
    expected = PRESETS["size-current"]
    for name, block_length in (("recursive-pilot-24", 24), ("recursive-pilot-48", 48)):
        preset = PRESETS[name]
        assert preset.d_model == expected.d_model
        assert preset.d_state == expected.d_state
        assert preset.n_layers == expected.n_layers
        assert preset.l_max == expected.l_max
        assert preset.batch_size == 8
        assert preset.recovery_corruption_mode == "recursive-block"
        assert preset.recursive_recovery_batch_fraction == 0.25
        assert preset.recursive_recovery_block_length == block_length

        smoke = replace(preset, d_model=32, d_state=8, n_layers=1, l_max=32)
        model = build_sequence_model(tokenizer, smoke, prediction_unit="triplet")
        assert model.output_vocab_size == 64


def test_recursive_training_step_is_finite():
    tokenizer = DnaTokenizer(include_n=False)
    codec = TripletCodec()
    sequence = "ACGT" * 8
    input_ids = torch.tensor([tokenizer.encode(sequence)], dtype=torch.long)
    target_ids = torch.full_like(input_ids, -100)
    for position in range(len(sequence) - 3):
        target_ids[0, position] = codec.encode(sequence[position + 1 : position + 4])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[:, 8 : len(sequence) - 3] = 1

    preset = replace(
        PRESETS["smoke"],
        d_model=32,
        d_state=8,
        n_layers=1,
        l_max=32,
        batch_size=1,
        recovery_corruption_mode="recursive-block",
        recursive_recovery_batch_fraction=1.0,
        recursive_recovery_block_length=6,
        homopolymer_loss_weight=0.0,
    )
    model = build_sequence_model(tokenizer, preset, prediction_unit="triplet")
    model.prediction_unit = "triplet"
    model.output_tokens = codec.triplets
    optimizer = build_optimizer(model, preset.lr, preset.weight_decay)
    loader = DataLoader(
        TensorDataset(input_ids, target_ids, attention_mask, loss_mask),
        batch_size=1,
    )

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        preset,
        tokenizer,
        0,
        "triplet",
        codec,
    )
    assert metrics["recursive_recovery_batches"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
