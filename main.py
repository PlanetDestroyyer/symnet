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
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded model state from {args.checkpoint}")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded raw weights from {args.checkpoint} (RMS not found)")
    
    trainer = PPOTrainer(
        model, 
        env, 
        device, 
        total_steps=args.steps, 
        obs_shape=obs_shape
    )
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'obs_rms_mean' in ckpt:
            trainer.obs_rms.mean = ckpt['obs_rms_mean']
            trainer.obs_rms.var = ckpt['obs_rms_var']
            trainer.obs_rms.count = ckpt['obs_rms_count']
            trainer.reward_rms.mean = ckpt['reward_rms_mean']
            trainer.reward_rms.var = ckpt['reward_rms_var']
            trainer.reward_rms.count = ckpt['reward_rms_count']
            trainer.global_step = ckpt.get('global_step', 0)
            print("Loaded RMS normalization stats and global_step.")
    
    print(f"Starting {args.steps} step config on {args.env}...")
    trainer.train(total_env_steps=args.steps, log_every=2000)
    
    # Save model_final.pt with RMS stats
    trainer._save_checkpoint()
    # Also explicitly save as model_final.pt for probe.py
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'obs_rms_mean': trainer.obs_rms.mean,
        'obs_rms_var': trainer.obs_rms.var,
        'obs_rms_count': trainer.obs_rms.count,
        'reward_rms_mean': trainer.reward_rms.mean,
        'reward_rms_var': trainer.reward_rms.var,
        'reward_rms_count': trainer.reward_rms.count,
        'global_step': trainer.global_step
    }, "checkpoints/model_final.pt")

if __name__ == "__main__":
    main()
