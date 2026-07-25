import io

import torch

from src.dna.training import build_optimizer
from src.models.s4.s4_kernel import SSKernelDiag
from src.models.s4.s4_layer import S4Layer
from src.models.s4_model import S4SequenceModel


def test_s4_diagonal():
    print("testing s4d kernel...")
    kernel = SSKernelDiag(d_model=4, d_state=16)
    K = kernel(64)
    state = kernel.default_state(2, torch.device("cpu"))
    y, new_state = kernel.step(torch.randn(2, 4), state)

    assert K.shape == (4, 64)
    assert y.shape == (2, 4)
    assert new_state.shape == state.shape
    assert torch.isfinite(K).all()
    print("s4d kernel works\n")


def test_layer_step_matches_convolution():
    print("testing recurrent step matches convolution...")
    for model_variant in ("legacy", "s4d_v2"):
        for length in (32, 512, 1024):
            torch.manual_seed(0)
            layer = S4Layer(
                d_model=4,
                d_state=16,
                dropout=0.0,
                kernel_type="diag",
                bidirectional=False,
                model_variant=model_variant,
            )
            layer.eval()

            x = torch.randn(2, length, 4)
            y_conv, _ = layer(x)

            state = layer.default_state(x.shape[0], x.device)
            ys = []
            for t in range(x.shape[1]):
                y_step, state = layer.step(x[:, t], state)
                ys.append(y_step)
            y_step = torch.stack(ys, dim=1)

            max_diff = (y_conv - y_step).abs().max().item()
            print(f"  {model_variant} length={length} max abs diff: {max_diff:.8f}")
            assert torch.allclose(y_conv, y_step, atol=1e-4, rtol=1e-4)
    print("recurrent step matches convolution\n")


def test_s4d_numerical_stability():
    print("testing s4d numerical stability...")
    torch.manual_seed(0)
    kernel = SSKernelDiag(d_model=8, d_state=32)
    dt, A, C = kernel._get_params()
    transition_magnitude = torch.exp(dt * A.real)
    generated_kernel = kernel(1024)

    assert torch.isfinite(dt).all()
    assert torch.isfinite(A).all()
    assert torch.isfinite(C).all()
    assert torch.isfinite(generated_kernel).all()
    assert (dt > 0).all()
    assert (A.real < 0).all()
    assert (transition_magnitude > 0).all()
    assert (transition_magnitude <= 1).all()
    print("s4d numerical stability works\n")


def test_causal_forward_pass():
    print("testing causal forward pass...")
    torch.manual_seed(0)
    model = S4SequenceModel(
        vocab_size=8,
        d_model=16,
        d_state=8,
        n_layers=2,
        dropout=0.0,
        kernel_type="diag",
        bidirectional=False,
        l_max=64,
        max_length=64,
    )
    model.eval()

    original = torch.randint(3, 8, (1, 64))
    changed = original.clone()
    changed[:, 32:] = torch.randint(3, 8, (1, 32))

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    assert torch.allclose(original_logits[:, :32], changed_logits[:, :32], atol=1e-5, rtol=1e-5)
    print("causal forward pass works\n")


def test_checkpoint_reload():
    print("testing checkpoint reload...")
    torch.manual_seed(0)
    config = {
        "vocab_size": 8,
        "d_model": 16,
        "d_state": 8,
        "n_layers": 2,
        "kernel_type": "diag",
        "l_max": 64,
    }
    model = S4SequenceModel(
        **config,
        dropout=0.0,
        bidirectional=False,
        max_length=64,
    )
    model.eval()
    input_ids = torch.randint(3, 8, (1, 64))

    with torch.no_grad():
        expected = model(input_ids)

    buffer = io.BytesIO()
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu")
    restored = S4SequenceModel(
        **checkpoint["model_config"],
        dropout=0.0,
        bidirectional=False,
        max_length=64,
    )
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restored.eval()

    with torch.no_grad():
        actual = restored(input_ids)

    assert torch.equal(expected, actual)
    print("checkpoint reload works\n")


def test_s4d_v2_parameters_and_optimizer():
    print("testing s4d v2 parameters and optimizer...")
    model = S4SequenceModel(
        vocab_size=8,
        d_model=16,
        d_state=8,
        n_layers=2,
        dropout=0.0,
        kernel_type="diag",
        model_variant="s4d_v2",
        bidirectional=False,
        l_max=64,
        max_length=64,
    )
    parameter_names = dict(model.named_parameters())
    assert "blocks.0.s4_layer.kernel.A_imag" in parameter_names
    assert "blocks.0.s4_layer.output_linear.0.weight" in parameter_names

    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.01, ssm_lr=1e-4)
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in parameter_names.items():
        group = parameter_groups[id(parameter)]
        if name.endswith(("kernel.log_dt", "kernel.log_A_real", "kernel.A_imag")):
            assert group["weight_decay"] == 0.0
            assert group["lr"] == 1e-4
        else:
            assert group["weight_decay"] == 0.01
            assert group["lr"] == 3e-4

    input_ids = torch.randint(3, 8, (1, 64))
    targets = torch.randint(3, 8, (1, 64))
    attention_mask = torch.ones_like(input_ids)
    loss = model.compute_loss(input_ids, targets, attention_mask, objective="autocomplete")
    loss.backward()
    assert parameter_names["blocks.0.s4_layer.kernel.A_imag"].grad is not None
    optimizer.step()
    print("s4d v2 parameters and optimizer work\n")


if __name__ == "__main__":
    test_s4_diagonal()
    test_layer_step_matches_convolution()
    test_s4d_numerical_stability()
    test_causal_forward_pass()
    test_checkpoint_reload()
    test_s4d_v2_parameters_and_optimizer()
    print("All active S4 tests passed.")
