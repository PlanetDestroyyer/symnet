"""
PPO Trainer for SymNet.

Key design:
  - Both agents share one nn.Module (SymNetModel)
  - Collect n_steps of experience, compute loss_A + loss_B, single backward + AdamW step
  - Gradient accumulation from BOTH agents before optimizer step (not alternating)
  - AdamW lr=3e-4, wd=0.01, cosine LR with 500-step warmup
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from symnet.models.symnet_model import SymNetModel
from symnet.env.gridworld import GridWorld
from symnet.rl.buffer import RolloutBuffer
from symnet.reward import COMM_PENALTY, TIMEOUT_PENALTY


# ────────────────────────────────────────────────────────────
# PPO hyper-parameters (spec §04)
# ────────────────────────────────────────────────────────────
PPO_CLIP_EPS    = 0.2
PPO_EPOCHS      = 4      # passes over each rollout buffer
PPO_BATCH_SIZE  = 256
ENTROPY_COEF    = 0.01
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5

GAMMA           = 0.99
GAE_LAMBDA      = 0.95
N_STEPS         = 2048   # steps per rollout

LR              = 3e-4
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 500


def _make_lr_lambda(warmup: int, total_steps: int) -> Callable[[int], float]:
    """Linear warmup then cosine annealing."""
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class PPOTrainer:
    """
    Dual-agent PPO trainer.

    Parameters
    ----------
    model      : shared SymNetModel instance
    env        : GridWorld instance
    device     : torch.device
    total_steps: total environment interaction steps for LR schedule
    use_wandb  : whether to log to wandb
    save_dir   : directory to save comm vector checkpoints
    save_every : save comm vectors every N global steps
    """

    def __init__(
        self,
        model: SymNetModel,
        env: GridWorld,
        device: torch.device,
        total_steps: int = 500_000,
        save_dir: str = "checkpoints",
        save_every: int = 10_000,
    ) -> None:
        self.model       = model.to(device)
        self.env         = env
        self.device      = device
        self.save_dir    = save_dir
        self.save_every  = save_every

        self._init_csv()

        self.optimizer = AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_make_lr_lambda(WARMUP_STEPS, total_steps),
        )

        self.buffer = RolloutBuffer(
            n_steps=N_STEPS,
            obs_shape=(75,),
            comm_dim=model.comm_dim,
            n_actions=model.n_actions,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        self.global_step          = 0
        self.episodes_done        = 0
        self.episode_rewards: list[float] = []
        self.episode_collisions: list[int] = []
        self.episode_successes: list[int] = []

        # Saved comm vectors + positions (for linear probe)
        self._comm_log: list[dict] = []

    # ──────────────────────────────────────────────────────
    # Collect
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_rollout(self) -> None:
        """Run env for N_STEPS steps, fill RolloutBuffer."""
        self.buffer.reset()

        (obs_a, obs_b), info = self.env.reset()
        comm_a = self.model.zero_comm(1, self.device)  # (1, comm_dim)
        comm_b = self.model.zero_comm(1, self.device)

        ep_reward     = 0.0
        ep_collisions = 0
        ep_steps      = 0

        for _ in range(N_STEPS):
            # Convert observations
            t_obs_a = torch.from_numpy(obs_a).float().unsqueeze(0).to(self.device)
            t_obs_b = torch.from_numpy(obs_b).float().unsqueeze(0).to(self.device)

            # Forward pass — agent A receives comm from B and vice-versa
            logits_a, new_comm_a, val_a = self.model(t_obs_a, comm_b)
            logits_b, new_comm_b, val_b = self.model(t_obs_b, comm_a)

            dist_a = torch.distributions.Categorical(logits=logits_a)
            dist_b = torch.distributions.Categorical(logits=logits_b)
            action_a = dist_a.sample()
            action_b = dist_b.sample()
            lp_a = dist_a.log_prob(action_a)
            lp_b = dist_b.log_prob(action_b)

            value = 0.5 * (val_a.item() + val_b.item())

            (obs_a_next, obs_b_next), env_reward, terminated, truncated, info = self.env.step(
                int(action_a.item()), int(action_b.item())
            )

            # Apply comm_penalty per step
            reward = env_reward + COMM_PENALTY

            # Apply timeout penalty when appropriate
            if truncated and not terminated:
                reward += TIMEOUT_PENALTY

            done = terminated or truncated

            ep_reward     += reward
            ep_steps      += 1
            ep_collisions += int(info["collision"])

            self.buffer.add(
                obs_a       = obs_a,
                obs_b       = obs_b,
                comm_a      = new_comm_a.squeeze(0).cpu().numpy(),
                comm_b      = new_comm_b.squeeze(0).cpu().numpy(),
                action_a    = int(action_a.item()),
                action_b    = int(action_b.item()),
                log_prob_a  = float(lp_a.item()),
                log_prob_b  = float(lp_b.item()),
                value       = value,
                reward      = reward,
                done        = done,
            )

            # Log comm vectors periodically
            if self.global_step % self.save_every == 0:
                self._comm_log.append({
                    "step":    self.global_step,
                    "comm_a":  new_comm_a.squeeze(0).cpu().numpy().copy(),
                    "comm_b":  new_comm_b.squeeze(0).cpu().numpy().copy(),
                    "pos_a":   info["pos_a"].copy(),
                    "pos_b":   info["pos_b"].copy(),
                    "goal_a":  info["goal_a"].copy(),
                    "goal_b":  info["goal_b"].copy(),
                })

            comm_a = new_comm_a
            comm_b = new_comm_b
            obs_a  = obs_a_next
            obs_b  = obs_b_next
            self.global_step += 1

            if done:
                self.episodes_done += 1
                self.episode_rewards.append(ep_reward)
                self.episode_collisions.append(ep_collisions)
                self.episode_successes.append(int(terminated))

                ep_reward     = 0.0
                ep_collisions = 0
                ep_steps      = 0

                (obs_a, obs_b), info = self.env.reset()
                comm_a = self.model.zero_comm(1, self.device)
                comm_b = self.model.zero_comm(1, self.device)

        # Compute last value for bootstrap
        t_obs_a = torch.from_numpy(obs_a).float().unsqueeze(0).to(self.device)
        _, _, last_val_a = self.model(t_obs_a, comm_b)
        self.buffer.compute_returns_and_advantages(last_val_a.item())

    # ──────────────────────────────────────────────────────
    # Update
    # ──────────────────────────────────────────────────────

    def _ppo_update(self) -> dict[str, float]:
        """Run PPO_EPOCHS passes over the buffer, return avg losses."""
        total_loss_val   = 0.0
        total_policy_val = 0.0
        total_entropy    = 0.0
        n_updates        = 0

        for _ in range(PPO_EPOCHS):
            for batch in self.buffer.get_batches(PPO_BATCH_SIZE, self.device):
                (
                    obs_a, obs_b,
                    comm_a, comm_b,
                    actions_a, actions_b,
                    old_lp_a, old_lp_b,
                    returns, advantages,
                ) = batch

                # ── Agent A forward ──────────────────────────────────────
                logits_a, _, val_a = self.model(obs_a, comm_b)
                dist_a     = torch.distributions.Categorical(logits=logits_a)
                lp_a       = dist_a.log_prob(actions_a)
                entropy_a  = dist_a.entropy().mean()

                ratio_a    = torch.exp(lp_a - old_lp_a)
                clipped_a  = torch.clamp(ratio_a, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
                policy_a   = -torch.min(ratio_a * advantages, clipped_a * advantages).mean()
                value_a    = F.mse_loss(val_a.squeeze(-1), returns)
                loss_a     = policy_a + VALUE_COEF * value_a - ENTROPY_COEF * entropy_a

                # ── Agent B forward ──────────────────────────────────────
                logits_b, _, val_b = self.model(obs_b, comm_a)
                dist_b     = torch.distributions.Categorical(logits=logits_b)
                lp_b       = dist_b.log_prob(actions_b)
                entropy_b  = dist_b.entropy().mean()

                ratio_b    = torch.exp(lp_b - old_lp_b)
                clipped_b  = torch.clamp(ratio_b, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
                policy_b   = -torch.min(ratio_b * advantages, clipped_b * advantages).mean()
                value_b    = F.mse_loss(val_b.squeeze(-1), returns)
                loss_b     = policy_b + VALUE_COEF * value_b - ENTROPY_COEF * entropy_b

                # ── Accumulate BOTH gradients before optimizer step ──────
                total_loss = loss_a + loss_b

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()

                total_loss_val   += total_loss.item()
                total_policy_val += (policy_a + policy_b).item()
                total_entropy    += (entropy_a + entropy_b).item() * 0.5
                n_updates        += 1

        n = max(1, n_updates)
        return {
            "loss":    total_loss_val   / n,
            "policy":  total_policy_val / n,
            "entropy": total_entropy    / n,
        }

    # ──────────────────────────────────────────────────────
    # Logging & Checkpointing
    # ──────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        import csv, os
        self.log_path = './logs/training.csv'
        os.makedirs('./logs', exist_ok=True)
        # Write header once
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'episode', 'reward', 'episode_length',
                'task_success', 'collision_rate', 
                'comm_l2_norm', 'actor_loss', 'critic_loss'
            ])

    def _log(self, losses: dict[str, float]) -> None:
        """Log metrics to CSV and stdout."""
        recent = 10
        avg_reward = (
            float(np.mean(self.episode_rewards[-recent:]))
            if self.episode_rewards else 0.0
        )
        collision_rate = (
            float(np.mean(self.episode_collisions[-recent:])) /
            max(1, self.env.max_steps)
            if self.episode_collisions else 0.0
        )
        success_rate = (
            float(np.mean(self.episode_successes[-recent:]))
            if self.episode_successes else 0.0
        )

        # Comm vector stats (from buffer)
        comm_a_norm = float(np.linalg.norm(self.buffer.comm_a, axis=1).mean())
        comm_b_norm = float(np.linalg.norm(self.buffer.comm_b, axis=1).mean())
        comm_norm_avg = (comm_a_norm + comm_b_norm) / 2.0
        
        # We don't track exact ep lengths per episode in the aggregates easily, so we use max_steps or estimated
        avg_ep_length = float(self.env.max_steps)

        print(
            f"[step {self.global_step:>8d}] "
            f"ep={self.episodes_done:>5d} "
            f"reward={avg_reward:>+7.2f} "
            f"success={success_rate:.2f} "
            f"collision={collision_rate:.3f} "
            f"|comm|={comm_a_norm:.3f}/{comm_b_norm:.3f} "
            f"loss={losses['loss']:.4f}"
        )

        import csv
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step,
                self.episodes_done,
                avg_reward,
                avg_ep_length,
                success_rate,
                collision_rate,
                comm_norm_avg,
                losses["policy"],
                losses["loss"] - losses["policy"],  # approx critic + entropy
            ])

    def _save_checkpoint(self) -> None:
        """Save model weights and comm vectors."""
        import os, pickle
        os.makedirs(self.save_dir, exist_ok=True)
        step = self.global_step
        torch.save(self.model.state_dict(), f"{self.save_dir}/model_{step}.pt")
        with open(f"{self.save_dir}/comm_vectors_{step}.pkl", "wb") as f:
            pickle.dump(self._comm_log, f)
        self._comm_log = []   # clear after saving

    # ──────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────

    def train(self, total_env_steps: int, log_every: int = N_STEPS) -> None:
        """
        Run training for `total_env_steps` environment steps.

        Collects N_STEPS at a time, runs PPO update, repeats.
        """
        updates_done = 0
        while self.global_step < total_env_steps:
            self._collect_rollout()
            losses = self._ppo_update()
            updates_done += 1

            if updates_done % max(1, log_every // N_STEPS) == 0:
                self._log(losses)

            # Save comm vectors every save_every steps
            if self._comm_log:
                self._save_checkpoint()

        print(f"\nTraining complete. {self.global_step} steps, {self.episodes_done} episodes.")
