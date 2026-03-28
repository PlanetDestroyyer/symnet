"""
RolloutBuffer — stores trajectories for both agents during PPO collection.

Stores `n_steps` of experience, then generates mini-batches for the PPO update.
Both agents share the same buffer (one reward stream, shared fate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """
    Fixed-size circular buffer for dual-agent PPO experience.

    Parameters
    ----------
    n_steps  : number of environment steps per rollout
    obs_shape : observation shape (channels, H, W)
    comm_dim  : communication vector dimension
    n_actions : number of discrete actions
    gamma     : discount factor for returns
    gae_lambda: GAE lambda for advantage estimation
    """

    n_steps:    int
    obs_shape:  tuple[int, ...]
    comm_dim:   int
    n_actions:  int
    gamma:      float = 0.99
    gae_lambda: float = 0.95

    # Filled by collect()
    obs_a:       np.ndarray = field(init=False)
    obs_b:       np.ndarray = field(init=False)
    comm_a:      np.ndarray = field(init=False)   # comm sent by A at this step
    comm_b:      np.ndarray = field(init=False)   # comm sent by B at this step
    actions_a:   np.ndarray = field(init=False)
    actions_b:   np.ndarray = field(init=False)
    log_probs_a: np.ndarray = field(init=False)
    log_probs_b: np.ndarray = field(init=False)
    values:      np.ndarray = field(init=False)   # average of V_a and V_b
    rewards:     np.ndarray = field(init=False)   # shared reward
    dones:       np.ndarray = field(init=False)

    # Computed after collection
    returns:    np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._ptr = 0
        self._full = False
        self._allocate()

    def _allocate(self) -> None:
        n = self.n_steps
        s = self.obs_shape
        c = self.comm_dim
        self.obs_a       = np.zeros((n, *s),  dtype=np.float32)
        self.obs_b       = np.zeros((n, *s),  dtype=np.float32)
        self.comm_a      = np.zeros((n, c),   dtype=np.float32)
        self.comm_b      = np.zeros((n, c),   dtype=np.float32)
        self.actions_a   = np.zeros(n,        dtype=np.int64)
        self.actions_b   = np.zeros(n,        dtype=np.int64)
        self.log_probs_a = np.zeros(n,        dtype=np.float32)
        self.log_probs_b = np.zeros(n,        dtype=np.float32)
        self.values      = np.zeros(n,        dtype=np.float32)
        self.rewards     = np.zeros(n,        dtype=np.float32)
        self.dones       = np.zeros(n,        dtype=np.float32)
        self.returns     = np.zeros(n,        dtype=np.float32)
        self.advantages  = np.zeros(n,        dtype=np.float32)

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    def add(
        self,
        obs_a:     np.ndarray,
        obs_b:     np.ndarray,
        comm_a:    np.ndarray,   # (comm_dim,)
        comm_b:    np.ndarray,
        action_a:  int,
        action_b:  int,
        log_prob_a: float,
        log_prob_b: float,
        value:     float,         # averaged value estimate
        reward:    float,
        done:      bool,
    ) -> None:
        i = self._ptr
        self.obs_a[i]       = obs_a
        self.obs_b[i]       = obs_b
        self.comm_a[i]      = comm_a
        self.comm_b[i]      = comm_b
        self.actions_a[i]   = action_a
        self.actions_b[i]   = action_b
        self.log_probs_a[i] = log_prob_a
        self.log_probs_b[i] = log_prob_b
        self.values[i]      = value
        self.rewards[i]     = reward
        self.dones[i]       = float(done)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True
            self._ptr = 0

    @property
    def is_full(self) -> bool:
        return self._full

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """
        Compute GAE(λ) advantages and discounted returns in-place.
        Call after filling n_steps of experience.
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int,
        device: torch.device,
    ):
        """Yield shuffled mini-batches as torch tensors."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.from_numpy(self.obs_a[idx]).to(device),
                torch.from_numpy(self.obs_b[idx]).to(device),
                torch.from_numpy(self.comm_a[idx]).to(device),
                torch.from_numpy(self.comm_b[idx]).to(device),
                torch.from_numpy(self.actions_a[idx]).to(device),
                torch.from_numpy(self.actions_b[idx]).to(device),
                torch.from_numpy(self.log_probs_a[idx]).to(device),
                torch.from_numpy(self.log_probs_b[idx]).to(device),
                torch.from_numpy(self.returns[idx]).to(device),
                torch.from_numpy(self.advantages[idx]).to(device),
            )
