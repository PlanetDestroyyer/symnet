"""
SymNetModel — the single neural network shared by both agents.

Architecture (per forward pass):
    obs (B, 3, 5, 5) → ConvEncoder → obs_emb (256)
    cat(obs_emb, comm_in) → Linear → (512) → MambaSSM → hidden (512)
    hidden → action_head → logits (4)
    hidden → comm_head  → tanh → comm_out (128)

Two agents call `model.forward()` independently with their own observations
and the other agent's previous comm vector. Weights are shared because it is
the same `nn.Module` instance. Differentiation comes purely from different inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_minimal import MambaSSM

# Architectural constants — match spec §04
OBS_CHANNELS  = 3
OBS_SIZE      = 5       # 5×5 patch
OBS_EMB_DIM   = 256
COMM_DIM      = 128
HIDDEN_DIM    = 512
N_ACTIONS     = 4       # up down left right


class SymNetModel(nn.Module):
    """
    Symmetric dual-agent model. One instance, two forward passes per step.

    Parameters
    ----------
    obs_channels : number of observation channels (default 3)
    obs_size     : side length of observation patch (default 5)
    obs_emb_dim  : dimension of encoded observation (default 256)
    comm_dim     : dimension of communication vector (default 128)
    hidden_dim   : Mamba hidden dimension (default 512)
    n_actions    : number of discrete actions (default 4)
    n_mamba_layers : depth of Mamba stack (default 2)
    """

    def __init__(
        self,
        obs_channels: int = OBS_CHANNELS,
        obs_size: int = OBS_SIZE,
        obs_emb_dim: int = OBS_EMB_DIM,
        comm_dim: int = COMM_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_actions: int = N_ACTIONS,
        n_mamba_layers: int = 2,
    ) -> None:
        super().__init__()

        self.comm_dim   = comm_dim
        self.hidden_dim = hidden_dim
        self.n_actions  = n_actions

        # ── Observation encoder ────────────────────────────────────────────
        # Input: (B, 3, 5, 5)
        # Conv2d: 3→32 channels, kernel 3×3 → output (B, 32, 3, 3)
        # Flatten → 288, Linear → obs_emb_dim
        conv_out_size = 32 * (obs_size - 2) * (obs_size - 2)  # k=3, no padding
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_size, obs_emb_dim),
            nn.ReLU(),
        )

        # ── Input projection: obs_emb + comm_in → hidden_dim ──────────────
        self.input_proj = nn.Linear(obs_emb_dim + comm_dim, hidden_dim)

        # ── Mamba SSM backbone ─────────────────────────────────────────────
        self.mamba = MambaSSM(
            d_model=hidden_dim,
            n_layers=n_mamba_layers,
        )

        # ── Output heads ──────────────────────────────────────────────────
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.comm_head   = nn.Linear(hidden_dim, comm_dim)

        # ── Value head for PPO ─────────────────────────────────────────────
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers, he init for conv."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        comm_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE agent.

        Parameters
        ----------
        obs     : (B, 75)                               — agent's local view (flattened)
        comm_in : (B, comm_dim)                         — other agent's last comm vector

        Returns
        -------
        action_logits : (B, n_actions)   — raw logits for action distribution
        comm_out      : (B, comm_dim)    — this agent's outgoing comm vector
        value         : (B, 1)          — state value estimate (for PPO)
        """
        # Reshape to (B, C, H, W) for Conv2d
        obs_spatial = obs.view(-1, 3, 5, 5)
        
        # Encode observation
        obs_emb = self.obs_encoder(obs_spatial)      # (B, obs_emb_dim)

        # Concatenate with incoming comm
        x = torch.cat([obs_emb, comm_in], dim=-1)   # (B, obs_emb_dim + comm_dim)
        x = F.relu(self.input_proj(x))               # (B, hidden_dim)

        # Mamba SSM — treat single step as sequence of length 1
        h = self.mamba(x)                            # (B, hidden_dim)

        # Output heads
        action_logits = self.action_head(h)          # (B, n_actions)
        comm_out      = torch.tanh(self.comm_head(h))  # (B, comm_dim) ∈ (-1,1)
        value         = self.value_head(h)           # (B, 1)

        return action_logits, comm_out, value

    def zero_comm(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Return a zeroed communication vector (for episode start)."""
        dev = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.comm_dim, device=dev)
