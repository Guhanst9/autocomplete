"""HiPPO initialization for S4 (Legendre scaled - LegS)."""
import math
import numpy as np
import torch
from typing import Tuple, Optional


def transition_legs(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """A, B transition matrices for HiPPO-LegS (Legendre scaled)."""
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    B = B.copy()
    return A, B


def nplr_legs(N: int, rank: int = 1, dtype: torch.dtype = torch.float32,
              B_clip: Optional[float] = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NPLR form of HiPPO-LegS.
    Returns w (N//2), P (rank, N//2), B (N//2), V (N, N//2) in complex form.
    """
    A, B = transition_legs(N)
    A = torch.as_tensor(A, dtype=dtype)
    B = torch.as_tensor(B[:, 0], dtype=dtype)

    # Rank correction for LegS
    P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)
    if rank > 1:
        P = torch.cat([P, torch.zeros(rank - 1, N, dtype=dtype)], dim=0)
    AP = A + P.T @ P

    # Diagonalize skew-symmetric part (eigh on Hermitian -1j*AP)
    W_re = torch.mean(torch.diagonal(AP)).item()
    APd = AP.double()
    H_mat = torch.complex(torch.zeros_like(APd), -APd)
    W_im, V = torch.linalg.eigh(H_mat)
    W_im = W_im.to(torch.cfloat)
    V = V.to(torch.cfloat)
    W = W_re + 1j * W_im

    _, idx = torch.sort(W.imag)
    W = W[idx]
    V = V[:, idx]
    V = V[:, : N // 2]
    W = W[: N // 2]

    V_inv = V.conj().T
    B = V_inv @ B.to(V)
    P = V_inv @ P.T.to(V)

    if B_clip is not None:
        B = B.real + 1j * torch.clamp(B.imag, min=-B_clip, max=B_clip)

    return W, P, B, V


def hippo_init(N: int, H: int, rank: int = 1,
               kernel_type: str = "diag") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Initialize S4 parameters for HiPPO-LegS.
    Returns (A, B, C) for diag or (A, P, B, V) for nplr.
    A: (H, N//2) complex for diag, (H, N//2) for nplr
    """
    if kernel_type == "diag":
        # S4D: diagonal init from HiPPO eigenvalues
        A, P, B, V = nplr_legs(N, rank=rank)
        A = -A.imag  # positive imag part for stability
        A = A.unsqueeze(0).expand(H, -1).clone()
        B = B.unsqueeze(0).expand(H, -1).clone()
        C = torch.randn(H, N // 2, dtype=torch.cfloat) * 0.5
        return A, B, C, None
    else:
        A, P, B, V = nplr_legs(N, rank=rank)
        A = A.unsqueeze(0).expand(H, -1).clone()
        P = P.unsqueeze(1).expand(-1, H, -1).clone()
        B = B.unsqueeze(0).expand(H, -1).clone()
        V = V.unsqueeze(0).expand(H, -1, -1).clone()
        return A, P, B, V
