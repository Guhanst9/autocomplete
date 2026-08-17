import _path  # noqa: F401

from unittest.mock import patch

import torch

from src.dna.checkpoint import build_model_from_config
from src.dna.data import DnaTokenizer, DnaWindowDataset
from src.dna.generation import generate_bases
from src.dna.prediction import TripletCodec
from src.dna.training import build_recovery_batch, homopolymer_end_loss
from src.models.s4_model import S4SequenceModel
from src.models.transformer_model import TransformerSequenceModel


def triplet_dataset() -> tuple[DnaWindowDataset, DnaTokenizer, TripletCodec]:
    tokenizer = DnaTokenizer()
    codec = TripletCodec()
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
        prediction_unit="triplet",
        triplet_codec=codec,
    )
    return dataset, tokenizer, codec


def test_triplet_codec():
    codec = TripletCodec()
    assert codec.vocab_size == 64
    assert codec.decode(codec.encode("AAA")) == "AAA"
    assert codec.decode(codec.encode("CGT")) == "CGT"
    assert codec.decode(codec.encode("TTT")) == "TTT"


def test_overlapping_triplet_targets():
    dataset, tokenizer, codec = triplet_dataset()
    with patch("src.dna.data.random.randint", return_value=4):
        input_ids, target_ids, attention_mask, loss_mask = dataset[0]

    assert input_ids.tolist() == tokenizer.encode("ACGTACGT")
    assert [codec.decode(value) for value in target_ids[:5].tolist()] == [
        "CGT",
        "GTA",
        "TAC",
        "ACG",
        "CGT",
    ]
    assert loss_mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert attention_mask.tolist() == [1] * 8


def test_triplet_recovery_uses_first_base_from_previous_position():
    dataset, tokenizer, codec = triplet_dataset()
    input_ids = torch.tensor([tokenizer.encode("ACGTACGT")])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]])
    logits = torch.zeros(1, 8, 64)
    logits[0, 2, codec.encode("AAA")] = 10.0
    logits[0, 3, codec.encode("CCC")] = 10.0
    logits[0, 4, codec.encode("GGG")] = 10.0

    corrupted, recovery_mask, changed, eligible = build_recovery_batch(
        input_ids,
        attention_mask,
        loss_mask,
        logits,
        tokenizer,
        1.0,
        prediction_unit="triplet",
        triplet_codec=codec,
    )
    assert eligible == 3
    assert changed == 3
    assert corrupted[0, 3].item() == tokenizer.vocab["A"]
    assert corrupted[0, 4].item() == tokenizer.vocab["C"]
    assert corrupted[0, 5].item() == tokenizer.vocab["G"]
    assert recovery_mask.tolist() == [[False, False, False, True, True, False, False, False]]


def test_triplet_homopolymer_loss_groups_first_base_classes():
    tokenizer = DnaTokenizer()
    codec = TripletCodec()
    a_id = tokenizer.vocab["A"]
    c_id = tokenizer.vocab["C"]
    input_ids = torch.tensor([[a_id] * 8])
    target_ids = torch.tensor([[codec.encode("CAA")] * 8])
    loss_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]])
    logits = torch.zeros(1, 8, 64, requires_grad=True)
    for triplet, class_id in codec.vocab.items():
        if triplet.startswith("A"):
            logits.data[0, 7, class_id] = 5.0

    loss, count = homopolymer_end_loss(
        logits,
        input_ids,
        target_ids,
        loss_mask,
        {tokenizer.vocab[base] for base in "ACGT"},
        8,
        prediction_unit="triplet",
        triplet_codec=codec,
        tokenizer=tokenizer,
    )
    assert count == 1
    assert loss.item() > 1.0
    loss.backward()
    assert torch.isfinite(logits.grad).all().item()


def tiny_s4() -> S4SequenceModel:
    model = S4SequenceModel(
        vocab_size=7,
        input_vocab_size=7,
        output_vocab_size=64,
        d_model=16,
        d_state=8,
        n_layers=1,
        dropout=0.0,
        kernel_type="diag",
        bidirectional=False,
        model_variant="s4d_v2",
        max_length=16,
    ).eval()
    model.prediction_unit = "triplet"
    model.output_tokens = TripletCodec().triplets
    return model


def tiny_transformer() -> TransformerSequenceModel:
    model = TransformerSequenceModel(
        vocab_size=7,
        input_vocab_size=7,
        output_vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        dropout=0.0,
        max_length=16,
    ).eval()
    model.prediction_unit = "triplet"
    model.output_tokens = TripletCodec().triplets
    return model


def test_triplet_causality_and_generation_lengths():
    tokenizer = DnaTokenizer()
    prompt = torch.tensor([tokenizer.encode("ACGTACGT")])
    for model in (tiny_s4(), tiny_transformer()):
        changed = prompt.clone()
        changed[:, 5:] = torch.tensor([tokenizer.encode("AAA")])
        first = model(prompt)
        second = model(changed)
        assert first.shape == (1, 8, 64)
        assert torch.allclose(first[:, :5], second[:, :5], atol=1e-5)
        for length in (1, 2, 3, 7):
            output = generate_bases(model, tokenizer, prompt, length)
            suffix = tokenizer.decode(output[0, prompt.size(1) :].tolist(), stop_at_eos=False)
            assert len(suffix) == length
            assert set(suffix) <= set("ACGT")


def test_triplet_s4_recurrent_logits_match_convolution():
    tokenizer = DnaTokenizer()
    model = tiny_s4()
    prompt = torch.tensor([tokenizer.encode("ACGTACGTACGT")])
    convolution_logits = model(prompt)[:, -1]

    states = [block.default_state(1, prompt.device) for block in model.blocks]
    for position in range(prompt.size(1)):
        hidden = model.embed(prompt[:, position])
        for index, block in enumerate(model.blocks):
            hidden, states[index] = block.step(hidden, states[index])
    recurrent_logits = model.lm_head(model.ln_f(hidden))
    assert torch.allclose(convolution_logits, recurrent_logits, atol=1e-4)


def test_triplet_transformer_cache_matches_full_forward():
    tokenizer = DnaTokenizer()
    codec = TripletCodec()
    model = tiny_transformer()
    prompt = torch.tensor([tokenizer.encode("ACGTAC")])
    cached = generate_bases(model, tokenizer, prompt, 9)

    generated = prompt.clone()
    for _ in range(3):
        context = generated[:, -model.max_length :]
        class_id = model(context)[:, -1].argmax(dim=-1)
        bases = codec.base_ids(tokenizer)[class_id]
        generated = torch.cat((generated, bases), dim=1)
    assert torch.equal(cached, generated)


def test_triplet_checkpoint_shape_and_base_default():
    tokenizer = DnaTokenizer()
    config = {
        "model_type": "transformer",
        "prediction_unit": "triplet",
        "output_vocab_size": 64,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_length": 16,
    }
    triplet_model = build_model_from_config(config, tokenizer)
    assert triplet_model.embed.num_embeddings == 7
    assert triplet_model.lm_head.out_features == 64
    assert triplet_model.lm_head.weight is not triplet_model.embed.weight

    base_model = build_model_from_config({**config, "prediction_unit": "base", "output_vocab_size": 7}, tokenizer)
    assert base_model.lm_head.weight is base_model.embed.weight


if __name__ == "__main__":
    test_triplet_codec()
    test_overlapping_triplet_targets()
    test_triplet_recovery_uses_first_base_from_previous_position()
    test_triplet_homopolymer_loss_groups_first_base_classes()
    test_triplet_causality_and_generation_lengths()
    test_triplet_s4_recurrent_logits_match_convolution()
    test_triplet_transformer_cache_matches_full_forward()
    test_triplet_checkpoint_shape_and_base_default()
    print("Triplet prediction tests passed.")
