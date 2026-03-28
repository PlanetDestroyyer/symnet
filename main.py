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
    args = parser.parse_args()

    device = get_device()
    
    if args.env == "gridworld":
        env = GridWorld(phase=3)
    elif args.env == "minecraft":
        from symnet.env.env_wrapper import MineDojoWrapper
        env = MineDojoWrapper()
    elif args.env == "minigrid":
        from symnet.env.minigrid_wrapper import MiniGridWrapper
        env = MiniGridWrapper()
        
    obs_shape = getattr(env, "obs_shape", (75,))
    obs_dim = int(np.prod(obs_shape))
        
    model = SymNetModel(obs_dim=obs_dim, comm_dim=128, hidden_dim=512, n_actions=4).to(device)
    
    trainer = PPOTrainer(
        model, 
        env, 
        device, 
        total_steps=1000, 
        obs_shape=obs_shape
    )
    
    print(f"Starting 1000 step smoke test on {args.env}...")
    trainer.train(total_env_steps=1000, log_every=200)

if __name__ == "__main__":
    main()
