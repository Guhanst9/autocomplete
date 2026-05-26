"""
S4 Layer and Block: pre-norm, FFT convolution, residual, optional bidirectional.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from src.models.s4.s4_kernel import SSKernelDiag, SSKernelNPLR
except ImportError:
    from .s4_kernel import SSKernelDiag, SSKernelNPLR


def fft_conv(u: torch.Tensor, K: torch.Tensor, dropout_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    FFT-based convolution.
    u: (batch, d_model, seq_len)
    K: (d_model, seq_len) or (d_model, L) with L >= seq_len
    """
    L = u.shape[-1]
    if K.shape[-1] < L:
        K = torch.nn.functional.pad(K, (0, L - K.shape[-1]))
    K = K[..., :L]
    n_fft = 2 * L
    u_f = torch.fft.rfft(u, n=n_fft, dim=-1)
    K_f = torch.fft.rfft(K, n=n_fft, dim=-1)
    y_f = u_f * K_f
    y = torch.fft.irfft(y_f, n=n_fft, dim=-1)[..., :L]
    if dropout_mask is not None:
        y = y * dropout_mask
    return y


class S4Layer(nn.Module):
    """
    Single S4 layer: LayerNorm -> S4 kernel (FFT conv or step) -> dropout -> residual.
    Supports both convolution mode (training) and recurrent step mode (inference).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        kernel_type: str = "diag",
        bidirectional: bool = False,
        l_max: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)

        kernel_cls = SSKernelDiag if kernel_type == "diag" else SSKernelNPLR
        kwargs = dict(d_model=d_model, d_state=d_state, l_max=l_max)
        if kernel_type == "nplr":
            kwargs["rank"] = 1
        self.kernel = kernel_cls(**kwargs)

        self.dropout = nn.Dropout(dropout)
        self.D_skip = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convolution mode for parallel training.
        x: (batch, seq_len, d_model)
        attention_mask: (batch, seq_len), 1 for valid, 0 for pad
        Returns: (batch, seq_len, d_model), state
        """
        residual = x
        x = self.norm(x)
        B, L, H = x.shape
        u = x.transpose(1, 2)

        if attention_mask is not None:
            u = u * attention_mask.unsqueeze(1).float()

        K = self.kernel(L)
        y = fft_conv(u, K)
        if self.bidirectional:
            K_rev = torch.flip(K, dims=[-1])
            y_rev = fft_conv(u, K_rev)
            y = y + y_rev

        y = y + u * self.D_skip.unsqueeze(-1)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        out = y + residual

        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).float()
        return out, state

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent step mode for autoregressive inference (O(1) per token).
        x: (batch, d_model) - single token embedding
        state: (batch, d_model, d_state//2) - recurrent state
        Returns: (batch, d_model), new_state
        """
        B, H = x.shape
        device = x.device
        
        # Initialize state if None
        if state is None:
            state = self.kernel.default_state(B, device)
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Recurrent step through kernel
        y, new_state = self.kernel.step(x_norm, state)
        
        # Add skip connection (D term)
        y = y + x_norm * self.D_skip
        
        # Apply dropout (only in training)
        y = self.dropout(y)
        
        # Residual connection
        out = y + x
        
        return out, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Get default initial state for recurrent mode."""
        return self.kernel.default_state(batch, device)


class S4Block(nn.Module):
    """
    Full S4 block: S4Layer -> LayerNorm -> MLP -> residual.
    Supports both convolution mode (training) and recurrent step mode (inference).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        kernel_type: str = "diag",
        bidirectional: bool = False,
        l_max: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        self.s4_layer = S4Layer(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            kernel_type=kernel_type,
            bidirectional=bidirectional,
            l_max=l_max,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convolution mode for parallel training."""
        x, new_state = self.s4_layer(x, state=state, attention_mask=attention_mask)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        return x, new_state

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent step mode for autoregressive inference (O(1) per token).
        x: (batch, d_model) - single token embedding
        state: recurrent state from S4Layer
        Returns: (batch, d_model), new_state
        """
        # S4 layer step
        x, new_state = self.s4_layer.step(x, state)
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual
        
        return x, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Get default initial state for recurrent mode."""
        return self.s4_layer.default_state(batch, device)
