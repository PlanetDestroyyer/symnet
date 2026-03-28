"""
Minimal pure-PyTorch Mamba implementation.

Based on the reference implementation by Albert Gu & Tri Dao:
  https://github.com/state-spaces/mamba

This version requires NO CUDA extensions — it runs on any device.
Selective SSM equations are identical to the official mamba-ssm library.

Architecture (one MambaBlock):
    x (B, L, d_model)
    → expand projection  → z, x_ssm    (B, L, d_inner)
    → depthwise conv1d   → x_conv
    → SSM (selective)    → y
    → silu(z) * y        → output
    → down projection    → (B, L, d_model)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaBlock(nn.Module):
    """
    Single Mamba block (selective SSM).

    Parameters
    ----------
    d_model   : model dimension (e.g. 512)
    d_state   : SSM state expansion factor (default 16)
    d_conv    : local convolution width (default 4)
    expand    : inner expansion factor (default 2)
    dt_rank   : rank of Δ projection; 'auto' → ceil(d_model/16)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = expand * d_model
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: in_proj splits into x and z
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Depthwise conv over sequence
        self.conv1d   = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Δ (dt) init
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # A (fixed spectrum), log-parameterised
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ---------- SSM scan (naive O(L·N), no CUDA kernel required) ----------

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run selective SSM over sequence x of shape (B, L, d_inner).
        Returns y of same shape.
        """
        B, L, d = x.shape  # noqa: N806

        A = -torch.exp(self.A_log.float())   # (d_inner, d_state)
        D = self.D.float()

        x_dbl = self.x_proj(x)               # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C = torch.split(          # noqa: N806
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))    # (B, L, d_inner)

        # Discretise A and B
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(
            torch.einsum("bld,dn->bldn", dt, A)
        )
        dB = torch.einsum("bld,bln->bldn", dt, B_ssm)  # (B, L, d_inner, d_state)

        # Selective scan (sequential — no CUDA kernel)
        h = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :].unsqueeze(-1)
            y_i = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y_i)
        y = torch.stack(ys, dim=1)           # (B, L, d_inner)
        y = y + x * D
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        returns : (B, L, d_model)
        """
        residual = x                          # skip connection outside the block
        xz = self.in_proj(x)                 # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)       # each (B, L, d_inner)

        # Conv1d expects (B, C, L)
        x_conv = rearrange(x_in, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[..., :x_in.shape[1]]   # trim causal padding
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z)                    # gate
        return self.out_proj(y)


class MambaSSM(nn.Module):
    """
    Stack of MambaBlocks with layer norm.

    Parameters
    ----------
    d_model    : token / embedding dimension
    n_layers   : number of blocks (default 2 for lightweight experiments)
    d_state    : SSM state dimension
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand),
            )
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model) or (B, d_model) — single step.
        Returns same shape.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)   # add sequence length 1

        for layer in self.layers:
            x = layer(x) + x    # residual

        x = self.norm_f(x)

        if squeeze:
            x = x.squeeze(1)
        return x
