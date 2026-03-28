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

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape=(3,64,64), comm_dim=128, d_model=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, *obs_shape)
        cnn_out = self.cnn(dummy).shape[1]
        
        self.proj = nn.Linear(cnn_out + comm_dim, d_model)
        self.proj_flat = nn.Linear(75 + comm_dim, d_model)

    def forward(self, obs, comm):
        if obs.dim() == 2:
            x = torch.cat([obs, comm], dim=-1)
            return F.relu(self.proj_flat(x))
        feat = self.cnn(obs)
        x = torch.cat([feat, comm], dim=-1)
        return F.relu(self.proj(x))


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
        obs_dim: int = 75,
        comm_dim: int = COMM_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_actions: int = N_ACTIONS,
        n_mamba_layers: int = 2,
    ) -> None:
        super().__init__()

        self.obs_dim    = obs_dim
        self.comm_dim   = comm_dim
        self.hidden_dim = hidden_dim
        self.n_actions  = n_actions

        # ── Encoder (BUG 6: CNN) ────────────────────────────────────────────────
        self.obs_encoder = CNNEncoder(obs_shape=(3, 64, 64), comm_dim=comm_dim, d_model=hidden_dim)

        # ── Mamba SSM backbone ─────────────────────────────────────────────
        self.mamba = MambaSSM(
            d_model=hidden_dim,
            n_layers=n_mamba_layers,
        )

        # ── Output heads ──────────────────────────────────────────────────
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.comm_head   = nn.Linear(hidden_dim, comm_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)

        # Auxiliary comm prediction head (BUG 1)
        self.comm_projection = nn.Linear(comm_dim, obs_dim)

        self.recurrent_state_A = None
        self.recurrent_state_B = None

        self._init_weights()

    def reset_states(self):
        self.recurrent_state_A = None
        self.recurrent_state_B = None

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
        x = self.obs_encoder(obs, comm_in)
        
        # Mamba requires (B, L, D) = (Batch, 1, hidden_dim)
        x_seq = x.unsqueeze(1)
        h = self.mamba(x_seq)
        h = h.squeeze(1)                  # (B, hidden_dim)

        # Output heads
        action_logits = self.action_head(h)          # (B, n_actions)
        comm_out      = torch.tanh(self.comm_head(h))  # (B, comm_dim) ∈ (-1,1)
        value         = self.value_head(h)           # (B, 1)

        return action_logits, comm_out, value

    def step(self, obs: torch.Tensor, comm_in: torch.Tensor, agent: str = 'A') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent inference mode for Minecraft deployment (single step).
        """
        x = self.obs_encoder(obs, comm_in)
        x = x.unsqueeze(1)
        x = self.mamba(x)
        x = x.squeeze(1)
        return self.action_head(x), self.comm_head(x)

    def zero_comm(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Return a zeroed communication vector (for episode start)."""
        dev = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.comm_dim, device=dev)
