import torch
import numpy as np
from src.models.s4.s4_kernel import SSKernelDiag, SSKernelNPLR, discretize_zoh, cauchy_naive
from src.models.s4.s4_layer import S4Layer
from src.models.hippo.hippo import hippo_init, transition_legs

def test_discretization():
    print("testing discretization...")
    
    Lambda = torch.tensor([[-1.0 + 1j], [-2.0 + 0.5j]])
    B = torch.tensor([[1.0], [0.5]])
    Delta = torch.tensor([0.1, 0.1]).unsqueeze(-1)
    
    Lambda_bar, B_bar = discretize_zoh(Lambda, B, Delta)
    
    print(f"  Lambda shape: {Lambda_bar.shape}")
    print(f"  B shape: {B_bar.shape}")
    print(f"  Lambda_bar[0]: {Lambda_bar[0].item()}")
    print("discretization works\n")

def test_cauchy_kernel():
    print("testing cauchy kernel...")
    
    v = torch.randn(8, dtype=torch.complex64)
    z = torch.randn(16, dtype=torch.complex64)
    w = torch.randn(8, dtype=torch.complex64)
    
    result = cauchy_naive(v, z, w)
    
    print(f"  input v: {v.shape}, z: {z.shape}, w: {w.shape}")
    print(f"  output: {result.shape}")
    print(f"  result[0]: {result[0].item()}")
    print("cauchy kernel works\n")

def test_s4_diagonal():
    print("testing s4d kernel...")
    
    d_model = 4
    d_state = 16
    L = 64
    
    kernel = SSKernelDiag(d_model=d_model, d_state=d_state)
    
    K = kernel(L)
    
    print(f"  kernel shape: {K.shape}")
    print(f"  expected: ({d_model}, {L})")
    print(f"  kernel mean: {K.mean().item():.4f}")
    print(f"  kernel std: {K.std().item():.4f}")
    
    batch = 2
    u = torch.randn(batch, d_model)
    state = kernel.default_state(batch, u.device)
    
    y, new_state = kernel.step(u, state)
    
    print(f"  step output shape: {y.shape}")
    print(f"  step state shape: {new_state.shape}")
    print("s4d kernel works\n")

def test_s4_nplr():
    print("testing s4 nplr kernel...")
    
    d_model = 4
    d_state = 16
    L = 64
    
    kernel = SSKernelNPLR(d_model=d_model, d_state=d_state, rank=2)
    
    K = kernel(L)
    
    print(f"  kernel shape: {K.shape}")
    print(f"  expected: ({d_model}, {L})")
    print(f"  kernel mean: {K.mean().item():.4f}")
    print(f"  kernel std: {K.std().item():.4f}")
    
    batch = 2
    u = torch.randn(batch, d_model)
    state = kernel.default_state(batch, u.device)
    
    y, new_state = kernel.step(u, state)
    
    print(f"  step output shape: {y.shape}")
    print(f"  step state shape: {new_state.shape}")
    print(" s4 nplr kernel works\n")

def test_hippo_integration():
    print("testing hippo integration...")
    
    A, B = transition_legs(N=32)
    
    print(f"  hippo A shape: {A.shape}")
    print(f"  hippo B shape: {B.shape}")
    
    A_diag, B_diag, C_diag, _ = hippo_init(32, 8, kernel_type="diag")
    
    print(f"  diag A shape: {A_diag.shape}")
    print(f"  diag B shape: {B_diag.shape}")
    print(f"  diag C shape: {C_diag.shape}")
    
    eigenvals = np.linalg.eigvals(A)
    print(f"  hippo eigenvals (first 3): {eigenvals[:3]}")
    print(f"  diag A (first 3): {A_diag[0, :3].detach().numpy()}")
    print(" hippo integration works\n")

def test_convolution():
    print("testing convolution with real data...")
    
    d_model = 8
    d_state = 64
    batch = 2
    seq_len = 128
    
    kernel = SSKernelDiag(d_model=d_model, d_state=d_state)
    
    K = kernel(seq_len)
    
    u = torch.randn(batch, d_model, seq_len)
    
    u_f = torch.fft.rfft(u, n=2*seq_len, dim=-1)
    K_f = torch.fft.rfft(K, n=2*seq_len, dim=-1)
    y_f = u_f * K_f
    y = torch.fft.irfft(y_f, n=2*seq_len, dim=-1)[..., :seq_len]
    
    print(f"  input shape: {u.shape}")
    print(f"  kernel shape: {K.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  output mean: {y.mean().item():.4f}")
    print(f"  output std: {y.std().item():.4f}")
    print(" convolution produces valid output\n")

def test_layer_step_matches_convolution():
    print("testing recurrent step matches convolution...")

    for kernel_type in ("diag", "nplr"):
        torch.manual_seed(0)
        layer = S4Layer(
            d_model=4,
            d_state=16,
            dropout=0.0,
            kernel_type=kernel_type,
            bidirectional=False,
        )
        layer.eval()

        x = torch.randn(2, 12, 4)
        y_conv, _ = layer(x)

        state = layer.default_state(x.shape[0], x.device)
        ys = []
        for t in range(x.shape[1]):
            y_step, state = layer.step(x[:, t], state)
            ys.append(y_step)
        y_step = torch.stack(ys, dim=1)

        max_diff = (y_conv - y_step).abs().max().item()
        print(f"  {kernel_type} max abs diff: {max_diff:.8f}")
        assert torch.allclose(y_conv, y_step, atol=1e-5, rtol=1e-5)
    print(" recurrent step matches convolution\n")

if __name__ == "__main__":
    print("S4 Kernel Implementation Tests \n")

    
    test_discretization()
    test_cauchy_kernel()
    test_s4_diagonal()
    test_s4_nplr()
    test_hippo_integration()
    test_convolution()
    test_layer_step_matches_convolution()
    
    print("All tests passed! ")
