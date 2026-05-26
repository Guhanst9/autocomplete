"""
S4 kernel layers: SSKernelDiag (S4D) and SSKernelNPLR.
Supports convolution mode (training) and step mode (inference).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from src.models.hippo.hippo import hippo_init
except ImportError:
    from models.hippo.hippo import hippo_init


def cauchy_naive(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """v: (..., N), z: (..., L), w: (..., N) -> (..., L)"""
    v = torch.cat([v, v.conj()], dim=-1)
    w = torch.cat([w, w.conj()], dim=-1)
    cauchy = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    return torch.sum(cauchy, dim=-2)


def log_vandermonde_naive(v: torch.Tensor, x: torch.Tensor, L: int) -> torch.Tensor:
    """v: (..., N), x: (..., N) -> (..., L)"""
    ar = torch.arange(L, device=x.device, dtype=x.dtype)
    vm = torch.exp(x.unsqueeze(-1) * ar)
    return 2 * (v * vm).sum(dim=-2).real


class SSKernelDiag(nn.Module):
    """
    S4D diagonal kernel. forward(L) returns convolution kernel; step(u, state) for recurrent.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        l_max: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.N = d_state
        self.H = d_model
        self.l_max = l_max
        N_half = d_state // 2

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A = -exp(log_A_real) + 1j * pi * arange (HiPPO-style)
        log_A_real = torch.log(0.5 * torch.ones(d_model, N_half))
        self.log_A_real = nn.Parameter(log_A_real)
        ar = torch.arange(N_half, dtype=torch.float).unsqueeze(0).expand(d_model, -1)
        self.register_buffer("A_imag", math.pi * ar)

        C = torch.randn(d_model, N_half, dtype=torch.cfloat)
        self.C_re = nn.Parameter(C.real)
        self.C_im = nn.Parameter(C.imag)

        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    def _get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        C = self.C_re + 1j * self.C_im
        return dt, A, C

    def forward(self, L: int) -> torch.Tensor:
        """Return kernel K: (H, L) for FFT convolution."""
        dt, A, C = self._get_params()
        dA = dt * A
        K = (torch.exp(dA) - 1.0) / A * C
        ar = torch.arange(L, device=A.device, dtype=torch.float32)
        K = K.unsqueeze(-1) * torch.exp(dA.unsqueeze(-1) * ar)
        K = 2 * K.sum(dim=1).real
        return K

    def _step_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt, A, C = self._get_params()
        B = torch.ones(self.H, self.N // 2, device=A.device, dtype=torch.cfloat)
        A_d = torch.exp(dt * A)
        B_d = (torch.exp(dt * A) - 1.0) / A * B
        return A_d, B_d, C


    def step(self, u: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # structured parameters derived from the kernel
        A_d, B_d, C = self._step_params()
        Bu = B_d * u.unsqueeze(-1)
        # hidden state updated on each input token
        new_state = A_d * state + Bu
        # runs per token — o(1) memory at inference
        y = (C * new_state).sum(dim=-1).real + self.D * u
        return y, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.H, self.N // 2, dtype=torch.cfloat, device=device)


class SSKernelNPLR(nn.Module):
    """
    S4 NPLR (normal-plus-low-rank) kernel. Uses Cauchy kernel for convolution.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        rank: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        l_max: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.N = d_state
        self.H = d_model
        self.rank = rank
        self.l_max = l_max
        N_half = d_state // 2

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        A, P, B, V = hippo_init(d_state, d_model, rank=rank, kernel_type="nplr")
        self.register_parameter("A", nn.Parameter(A))
        self.register_parameter("P", nn.Parameter(P))
        self.register_parameter("B", nn.Parameter(B))
        self.V = V
        C = torch.randn(d_model, N_half, dtype=torch.cfloat)
        self.C_re = nn.Parameter(C.real)
        self.C_im = nn.Parameter(C.imag)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, L: int) -> torch.Tensor:
        # compute time step from learned log scale
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = self.A
        C = self.C_re + 1j * self.C_im
        # discretize continuous-time state space
        dA = dt * A
        K = (torch.exp(dA) - 1.0) / A * C
        # build convolution kernel across sequence length l
        ar = torch.arange(L, device=A.device, dtype=torch.float32)
        K = K.unsqueeze(-1) * torch.exp(dA.unsqueeze(-1) * ar)
        # sum over state dim -> ready for fft convolution
        return 2 * K.sum(dim=1).real

    def _step_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt, A, C = torch.exp(self.log_dt).unsqueeze(-1), self.A, self.C_re + 1j * self.C_im
        B = self.B
        A_d = torch.exp(dt * A)
        B_d = (torch.exp(dt * A) - 1.0) / A * B
        return A_d, B_d, C

    def step(self, u: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A_d, B_d, C = self._step_params()
        Bu = B_d * u.unsqueeze(-1)
        new_state = A_d * state + Bu
        y = (C * new_state).sum(dim=-1).real + self.D * u
        return y, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.H, self.N // 2, dtype=torch.cfloat, device=device)
