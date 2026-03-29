import argparse
import torch
import numpy as np
from symnet.env.gridworld import GridWorld
from symnet.models.symnet_model import SymNetModel
from symnet.rl.ppo import PPOTrainer
from symnet.utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gridworld", choices=["gridworld", "minecraft", "minigrid"])
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    if args.mode == 'probe':
        print("Running linear probe...")
        from probe import run_probe
        run_probe(comm_dir="./comm_logs", model_path="checkpoints/model_final.pt")
        return

    device = get_device()
    
    if args.env == "gridworld":
        env = GridWorld(phase=args.phase)
    elif args.env == "minecraft":
        from symnet.env.env_wrapper import MineDojoWrapper
        env = MineDojoWrapper()
    elif args.env == "minigrid":
        from symnet.env.minigrid_wrapper import MiniGridWrapper
        env = MiniGridWrapper()
        
    obs_shape = getattr(env, "obs_shape", (75,))
    obs_dim = int(np.prod(obs_shape))
        
    model = SymNetModel(obs_dim=obs_dim, comm_dim=128, hidden_dim=512, n_actions=4).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    trainer = PPOTrainer(
        model, 
        env, 
        device, 
        total_steps=args.steps, 
        obs_shape=obs_shape
    )
    
    print(f"Starting {args.steps} step config on {args.env}...")
    trainer.train(total_env_steps=args.steps, log_every=2000)
    
    # Save model_final.pt
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model_final.pt")

if __name__ == "__main__":
    main()
