"""
train.py — main SymNet training entrypoint.

Usage:
    python train.py [options]

Examples:
    # Quick smoke test (no wandb)
    python train.py --steps 5000 --no-wandb

    # Phase 3 training with wandb
    python train.py --steps 500000 --phase 3 --wandb

    # Full curriculum
    python train.py --steps 1000000 --curriculum
"""

from __future__ import annotations

import argparse
import os

import torch

from symnet.env.gridworld import GridWorld
from symnet.models.symnet_model import SymNetModel
from symnet.rl.ppo import PPOTrainer
from symnet.utils import set_seed, get_device, make_checkpoint_dir


# ────────────────────────────────────────────────────────────
# Curriculum phase definitions
# ────────────────────────────────────────────────────────────
PHASES = {
    1: {"steps":  50_000, "grid_size": 8, "obstacles": 8, "label": "single agent"},
    2: {"steps": 100_000, "grid_size": 8, "obstacles": 8, "label": "both agents independent"},
    3: {"steps": 350_000, "grid_size": 8, "obstacles": 8, "label": "cooperative task"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SymNet")
    p.add_argument("--steps",      type=int,  default=500_000, help="Total env steps")
    p.add_argument("--grid-size",  type=int,  default=8,       help="Grid side length")
    p.add_argument("--obstacles",  type=int,  default=8,       help="Number of obstacles")
    p.add_argument("--max-steps",  type=int,  default=200,     help="Max steps per episode")
    p.add_argument("--seed",       type=int,  default=42,      help="Random seed")
    p.add_argument("--save-dir",   type=str,  default="checkpoints")
    p.add_argument("--save-every", type=int,  default=10_000)
    p.add_argument("--phase",      type=int,  default=None,
                   help="Run a specific training phase (1-3)")
    p.add_argument("--curriculum", action="store_true",
                   help="Run the full 3-phase curriculum in sequence")
    p.add_argument("--hidden-dim", type=int,  default=512)
    p.add_argument("--comm-dim",   type=int,  default=128)
    p.add_argument("--n-mamba-layers", type=int, default=2)
    return p.parse_args()


def build_env(grid_size: int, obstacles: int, max_steps: int, seed: int, phase: int) -> GridWorld:
    return GridWorld(
        grid_size=grid_size,
        num_obstacles=obstacles,
        max_steps=max_steps,
        seed=seed,
        phase=phase,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> SymNetModel:
    return SymNetModel(
        hidden_dim=args.hidden_dim,
        comm_dim=args.comm_dim,
        n_mamba_layers=args.n_mamba_layers,
    ).to(device)


def run_phase(
    model: SymNetModel,
    phase: int,
    steps: int,
    grid_size: int,
    obstacles: int,
    max_steps: int,
    args: argparse.Namespace,
    device: torch.device,
    label: str = "",
) -> PPOTrainer:
    print(f"\n{'='*60}")
    print(f"  Phase: {label}   steps={steps}  grid={grid_size}  obstacles={obstacles}")
    print(f"{'='*60}")

    env = build_env(grid_size, obstacles, max_steps, args.seed, phase=phase)
    trainer = PPOTrainer(
        model=model,
        env=env,
        device=device,
        total_steps=steps,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )
    trainer.train(total_env_steps=steps)
    return trainer


def main() -> None:
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)
    make_checkpoint_dir(args.save_dir)

    print(f"SymNet Training")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.curriculum:
        # Run all 3 phases in sequence, carrying weights forward
        for phase_id in range(1, 4):
            cfg = PHASES[phase_id]
            run_phase(
                model     = model,
                phase     = phase_id,
                steps     = cfg["steps"],
                grid_size = cfg["grid_size"],
                obstacles = cfg["obstacles"],
                max_steps = 200,
                args      = args,
                device    = device,
                label     = f"{phase_id} — {cfg['label']}",
            )
        # Save final model
        final_path = os.path.join(args.save_dir, "model_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"\nFinal model saved to {final_path}")

    elif args.phase is not None:
        cfg = PHASES[args.phase]
        run_phase(
            model     = model,
            phase     = args.phase,
            steps     = cfg["steps"],
            grid_size = cfg["grid_size"],
            obstacles = cfg["obstacles"],
            max_steps = 200,
            args      = args,
            device    = device,
            label     = f"{args.phase} — {cfg['label']}",
        )

    else:
        # Single custom run - default to phase 3 (cooperative)
        run_phase(
            model     = model,
            phase     = 3,
            steps     = args.steps,
            grid_size = args.grid_size,
            obstacles = args.obstacles,
            max_steps = args.max_steps,
            args      = args,
            device    = device,
            label     = "custom",
        )


if __name__ == "__main__":
    main()
