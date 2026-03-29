"""
probe.py — Linear Probe and Interventions for SymNet

1. Probes comm vectors to predict: position, goal, direction.
2. Runs intervention tests: normal vs zero vs gaussian comm_vectors.
3. Saves report to ./probe_results/report.json
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from symnet.env.gridworld import GridWorld
from symnet.models.symnet_model import SymNetModel
from symnet.utils import get_device

def load_comm_vectors_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

def compute_direction(p1: np.ndarray, p2: np.ndarray) -> int:
    """Return 8-way compass direction from p1 to p2."""
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    angle = np.arctan2(dy, dx)
    return int((angle / np.pi + 1.0) * 4) % 8

def run_single_probe(X: np.ndarray, y: np.ndarray) -> dict:
    if len(X) == 0:
        return {"accuracy": 0.0, "chance": 0.0}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    chance = 1.0 / max(1, len(np.unique(y)))
    return {"accuracy": float(acc), "chance": float(chance)}

def run_probes(records: list[dict], grid_size: int = 8) -> dict:
    # We will probe comm_b to predict B's properties (which A receives)
    # 1. Position B
    # 2. Goal B
    # 3. Direction from A to Goal B
    
    X, y_pos, y_goal, y_dir = [], [], [], []
    for rec in records:
        comm_b = rec["comm_b"]
        pos_a  = rec["pos_a"]
        pos_b  = rec["pos_b"]
        goal_b = rec["goal_b"]

        X.append(comm_b)
        y_pos.append(int(pos_b[0] * grid_size + pos_b[1]))
        y_goal.append(int(goal_b[0] * grid_size + goal_b[1]))
        y_dir.append(compute_direction(pos_a, goal_b))

    X = np.array(X, dtype=np.float32)
    
    print("Running Position Probe...")
    res_pos = run_single_probe(X, np.array(y_pos, dtype=np.int64))
    print("Running Goal Probe...")
    res_goal = run_single_probe(X, np.array(y_goal, dtype=np.int64))
    print("Running Direction Probe...")
    res_dir = run_single_probe(X, np.array(y_dir, dtype=np.int64))

    return {
        "position": res_pos,
        "goal": res_goal,
        "direction": res_dir,
    }

@torch.no_grad()
def run_intervention_test(
    model: SymNetModel,
    env: GridWorld,
    device: torch.device,
    mode: str,
    n_episodes: int = 100,
    obs_rms: any = None,
) -> dict:
    model.eval()
    successes = 0
    total_reward = 0.0

    for _ in range(n_episodes):
        (obs_a, obs_b), info = env.reset()
        comm_a = model.zero_comm(1, device)
        comm_b = model.zero_comm(1, device)
        
        ep_reward = 0.0
        done = False
        
        while not done:
            # Normalize if obs_rms is provided
            if obs_rms is not None:
                norm_obs_a = obs_rms.normalize(obs_a)
                norm_obs_b = obs_rms.normalize(obs_b)
            else:
                norm_obs_a = obs_a
                norm_obs_b = obs_b
                
            t_obs_a = torch.from_numpy(norm_obs_a).float().unsqueeze(0).to(device)
            t_obs_b = torch.from_numpy(norm_obs_b).float().unsqueeze(0).to(device)

            # Intervene
            if mode == "zero":
                used_comm_a = model.zero_comm(1, device)
                used_comm_b = model.zero_comm(1, device)
            elif mode == "gaussian":
                used_comm_a = torch.randn_like(comm_a)
                used_comm_b = torch.randn_like(comm_b)
            else:
                used_comm_a = comm_a
                used_comm_b = comm_b

            logits_a, new_comm_a, _ = model(t_obs_a, used_comm_b)
            logits_b, new_comm_b, _ = model(t_obs_b, used_comm_a)

            dist_a = torch.distributions.Categorical(logits=logits_a)
            dist_b = torch.distributions.Categorical(logits=logits_b)
            action_a = dist_a.sample().item()
            action_b = dist_b.sample().item()

            (obs_a, obs_b), reward, terminated, truncated, info = env.step(action_a, action_b)
            done = terminated or truncated
            ep_reward += reward
            comm_a, comm_b = new_comm_a, new_comm_b

        total_reward += ep_reward
        if info.get("both_at_goal", False):
            successes += 1

    return {
        "success_rate": float(successes / max(1, n_episodes)),
        "avg_reward": float(total_reward / max(1, n_episodes)),
    }

def run_probe(comm_dir="comm_logs", model_path="checkpoints/model_final.pt") -> None:
    # 1. Linear Probes
    print("Loading comm vectors...")
    records = []
    pattern = os.path.join(comm_dir, "comm_*.pt")
    for fp in sorted(glob.glob(pattern)):
        ckpt = load_comm_vectors_pt(fp)
        for i in range(len(ckpt['comm_B'])):
            records.append({
                "comm_a": ckpt['comm_A'][i].numpy(),
                "comm_b": ckpt['comm_B'][i].numpy(),
                "pos_a": ckpt['pos_A'][i],
                "pos_b": ckpt['pos_B'][i],
                "goal_a": ckpt['goal_A'][i],
                "goal_b": ckpt['goal_B'][i],
            })
    
    probe_results = {}
    if records:
        print(f"Loaded {len(records)} records. Running probes...")
        probe_results = run_probes(records)
    else:
        print("No records found to probe.")

    # 2. Interventions
    interv_results = {}
    device = get_device()
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} for interventions...")
        ckpt = torch.load(model_path, map_location=device)
        model = SymNetModel().to(device)
        
        from symnet.rl.rms import RunningMeanStd
        obs_rms = RunningMeanStd(shape=(75,))
        
        if isinstance(ckpt, dict) and 'obs_rms_mean' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            obs_rms.mean = ckpt['obs_rms_mean']
            obs_rms.var = ckpt['obs_rms_var']
            obs_rms.count = ckpt['obs_rms_count']
            print("Loaded observation normalization stats from checkpoint.")
        else:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            print("WARNING: No normalization stats in checkpoint. Running 5000-step warmup with TRAINED policy...")
            # Fallback: estimate RMS from the environment using the policy itself!
            env_warmup = GridWorld(grid_size=8, phase=3)
            model.eval()
            (o_a, o_b), _ = env_warmup.reset()
            c_a = model.zero_comm(1, device)
            c_b = model.zero_comm(1, device)
            for _ in range(5000):
                obs_rms.update(o_a)
                obs_rms.update(o_b)
                # Use current RMS to move
                no_a = obs_rms.normalize(o_a)
                no_b = obs_rms.normalize(o_b)
                t_o_a = torch.from_numpy(no_a).float().unsqueeze(0).to(device)
                t_o_b = torch.from_numpy(no_b).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    l_a, nc_a, _ = model(t_o_a, c_b)
                    l_b, nc_b, _ = model(t_o_b, c_a)
                
                # Sample
                a_a = torch.distributions.Categorical(logits=l_a).sample().item()
                a_b = torch.distributions.Categorical(logits=l_b).sample().item()
                
                (o_a_next, o_b_next), _, term, trunc, _ = env_warmup.step(a_a, a_b)
                if term or trunc:
                    (o_a, o_b), _ = env_warmup.reset()
                    c_a = model.zero_comm(1, device)
                    c_b = model.zero_comm(1, device)
                else:
                    o_a, o_b = o_a_next, o_b_next
                    c_a, c_b = nc_a, nc_b
        
        env = GridWorld(grid_size=8, phase=3)
        
        for mode in ["normal", "zero", "gaussian"]:
            print(f"Running intervention: {mode}")
            res = run_intervention_test(model, env, device, mode, n_episodes=100, obs_rms=obs_rms)
            interv_results[mode] = res
    else:
        print(f"Model not found at {model_path}; skipping interventions.")

    # 3. Report
    report = {
        "probes": probe_results,
        "interventions": interv_results,
    }

    os.makedirs("probe_results", exist_ok=True)
    out_path = "probe_results/report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    import sys
    c_dir = sys.argv[1] if len(sys.argv) > 1 else 'comm_logs'
    m_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/model_final.pt'
    run_probe(c_dir, m_path)
