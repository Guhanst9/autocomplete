from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from src.models.s4.s4_kernel import SSKernelDiag, SSKernelNPLR
except ImportError:
    from .s4_kernel import SSKernelDiag, SSKernelNPLR


def fft_conv(u: torch.Tensor, K: torch.Tensor, dropout_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        # recurrent path used for autoregressive generation
        B, H = x.shape
        device = x.device
        
        if state is None:
            state = self.kernel.default_state(B, device)
        
        x_norm = self.norm(x)
        
        y, new_state = self.kernel.step(x_norm, state)
        
        y = y + x_norm * self.D_skip
        
        y = self.dropout(y)
        
        out = y + x
        
        return out, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.kernel.default_state(batch, device)


class S4Block(nn.Module):
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
        x, new_state = self.s4_layer.step(x, state)
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual
        
        return x, new_state

    def default_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.s4_layer.default_state(batch, device)
