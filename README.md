# SymNet

A symmetric dual-agent reinforcement learning system grounded in an emergent continuous communication language. 

## About The Project

SymNet explores how cooperative agents can evolve a communication protocol from scratch without predefined vocabularies or asymmetric hardcoded roles (e.g. "Speaker" vs "Listener"). 

**Key Features:**
* **Symmetric Architecture**: Both Agent A and Agent B share the exact same `SymNetModel` (a single PyTorch `nn.Module` in memory). They only differentiate their behavior based on their independent local grid observations and the incoming communication vectors.
* **Continuous Communication**: Agents communicate via unconstrained, continuous 128-dimensional float vectors computed at every timestep.
* **Mamba SSM Backbone**: Uses a modern Selective State-Space Model (`mamba-minimal`) as the sequence processor, running entirely in native PyTorch. This avoids difficult CUDA C++ extension compilations and **automatically runs on Kaggle/Colab GPUs** out of the box!
* **PPO with Gradient Accumulation**: Training is powered by a Proximal Policy Optimization (PPO) trainer that computes and accumulates loss from *both* agents' parallel execution graphs before executing a single shared AdamW optimizer step.

## Setup & Dependencies

Dependencies are tracked in `pyproject.toml` and `requirements.txt`. Native PyTorch handles the matrix operations natively, making this fully compatible with CPU, CUDA, and Apple Silicon.

```bash
# Using uv (recommended)
uv sync --extra dev

# Or pip
pip install -r requirements.txt
```

## Running the Training Curriculum

SymNet implements a curriculum learning approach where environment complexity scales up as agents learn basic navigation.

```bash
# Run the full 3-phase curriculum (Single Agent -> Independent -> Cooperative)
uv run python train.py --curriculum

# Run a quick 5k step smoke test
uv run python train.py --steps 5000
```
Metrics and checkpoints are saved under `./checkpoints/` and logged to `./logs/training.csv`.

## Inspecting Communication

The `probe.py` script validates whether the continuous 128-dim communication embeddings intrinsically contain real spatial meaning. 

```bash
# Probe saved communication vectors to predict the other agent's position/goal/direction
uv run python probe.py --comm-dir checkpoints/
```

This runs a suite of supervised learning probes (predicting exact target coordinates from the other agent's pure communication vector) and an intervention test to calculate the lift over random chance!
