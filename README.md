# Husky RL — Reinforcement Learning for Robotic Pick-and-Place

A final-year project implementing and comparing three deep reinforcement learning algorithms — **PPO**, **SAC**, and **TQC** — on a simulated Husky robot performing a multi-phase pick-and-place task in PyBullet.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Task Description](#task-description)
- [Observation and Action Space](#observation-and-action-space)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Running a Trained Agent](#running-a-trained-agent)
  - [Evaluating and Comparing Algorithms](#evaluating-and-comparing-algorithms)
- [Checkpoints](#checkpoints)
- [Outputs](#outputs)
- [Extending the Project](#extending-the-project)

---

## Project Overview

The Husky robot (simulated in [PyBullet](https://pybullet.org/)) is equipped with a front-facing camera and a forklift mast. Its goal is to locate a small green cylinder, pick it up using a magnetic gripper, navigate to a red hollow box, and drop the cylinder inside it — all using only raw camera pixel statistics as sensor input.

Three discrete-action RL algorithms are trained and compared:

| Algorithm | Type | Key Trait |
|---|---|---|
| **PPO** | On-policy | Clipped surrogate objective, GAE, early KL stopping |
| **SAC** | Off-policy | Entropy-regularised, twin-critic, automatic temperature |
| **TQC** | Off-policy | Distributional critics (quantile regression), truncated Q |

---

## Task Description

The task is broken into **6 sequential phases**. The agent must complete them in order:

| Phase | ID | Description |
|---|---|---|
| `find-G` | 0 | Spin/explore until the green cylinder appears in the camera |
| `appr-G` | 1 | Drive forward and centre the green cylinder in the frame |
| `pickup` | 2 | **Automated** — lower fork, attach magnet, lift cylinder |
| `find-R` | 3 | Spin/explore until the red box appears in the camera |
| `appr-R` | 4 | Drive forward and centre the red box in the frame |
| `drop` | 5 | **Automated** — release magnet, reverse, lower fork, check success |

Phases `pickup` and `drop` are fully automated; the agent only controls navigation (phases 0, 1, 3, 4).

**Episode ends when:**
- The drop phase completes (success or miss), or
- 700 steps elapse (timeout)

**Win condition:** The green cylinder physically lands inside the red hollow box after release.

---

## Observation and Action Space

### Observation (10 floats)

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `green_cx` | [0, 1] | Horizontal centre of green object in frame |
| 1 | `green_cy` | [0, 1] | Vertical centre of green object in frame |
| 2 | `green_area` | [0, 1] | Fraction of pixels occupied by green object |
| 3 | `green_vis` | {0, 1} | 1 if green object is visible |
| 4 | `red_cx` | [0, 1] | Horizontal centre of red object in frame |
| 5 | `red_cy` | [0, 1] | Vertical centre of red object in frame |
| 6 | `red_area` | [0, 1] | Fraction of pixels occupied by red object |
| 7 | `red_vis` | {0, 1} | 1 if red object is visible |
| 8 | `lift_norm` | [0, 1] | Normalised fork lift position |
| 9 | `phase_norm` | [0, 1] | Normalised current task phase |

### Action Space (5 discrete actions)

| Action | Name | Description |
|---|---|---|
| 0 | `spin-L` | Spin in place — left |
| 1 | `spin-R` | Spin in place — right |
| 2 | `fwd` | Drive straight forward |
| 3 | `fwd+L` | Drive forward with left steer |
| 4 | `fwd+R` | Drive forward with right steer |

An **action mask** is applied: spinning away from a visible target is blocked at inference time to prevent trivially bad actions.

---

## Algorithms

### PPO (Proximal Policy Optimisation)
- On-policy actor-critic with separate actor and critic networks
- Collects a rollout buffer of 4096 steps, then performs 10 mini-batch SGD epochs
- Uses clipped surrogate loss, generalised advantage estimation (GAE), and value loss clipping
- KL divergence early stopping prevents large policy updates

### SAC (Soft Actor-Critic, Discrete)
- Off-policy actor with twin Q-critics and experience replay (100k buffer)
- Entropy-regularised objective with automatic temperature (`alpha`) tuning
- Targets maximum-entropy exploration while learning a deterministic greedy policy

### TQC (Truncated Quantile Critics)
- Extends SAC with distributional critics — each critic models Q-value distributions as N quantiles instead of a scalar
- Uses 5 critic networks × 25 quantiles; drops the top 2 quantiles per critic to reduce overestimation bias
- Same actor structure as SAC

All three algorithms share the same neural network hidden size (256), learning rate (3e-4), and discount factor (γ = 0.99).

---

## Project Structure

```
8_sem/
├── train.py              # Entry point for training and inference
├── evaluate.py           # Multi-algorithm evaluation and plotting
├── requirements.txt      # Python dependencies
├── assets/
│   └── forklift_mast.urdf    # URDF model for the forklift attachment
├── checkpoints/          # Saved model weights (.pth files)
├── outputs/              # Evaluation plots
└── husky_rl/
    ├── __init__.py
    ├── config.py         # All hyperparameters and environment constants
    ├── environment.py    # HuskyTask2Env — PyBullet simulation
    ├── models.py         # Neural network architectures (ActorCritic, SACDiscreteActor, TQCCritic)
    └── agents/
        ├── __init__.py
        ├── ppo.py        # PPO training loop and rollout buffer
        ├── sac.py        # SAC training loop and replay buffer
        └── tqc.py        # TQC training loop and distributional critics
```

---

## Installation

**Requirements:** Python 3.8+, pip

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:
- `numpy`
- `torch`
- `pybullet`
- `matplotlib`

> **GPU training:** PyTorch will automatically use CUDA if available. Install the appropriate CUDA-enabled torch version from [pytorch.org](https://pytorch.org) if needed.

---

## Usage

### Training

Train with the default algorithm (PPO):

```bash
python train.py
```

Train with a specific algorithm:

```bash
python train.py --algo ppo   # Proximal Policy Optimisation
python train.py --algo sac   # Soft Actor-Critic
python train.py --algo tqc   # Truncated Quantile Critics
```

Training options:

| Flag | Default | Description |
|---|---|---|
| `--algo` | `ppo` | Algorithm to use: `ppo`, `sac`, or `tqc` |

Training prints a live log showing episode reward, rolling average, win rate, current phase reached, and best average reward. Checkpoints are saved automatically to `checkpoints/`.

**Early stopping:** Training ends when the 50-episode rolling win rate reaches 85%.

---

### Running a Trained Agent

Launch a GUI visualisation of a trained agent:

```bash
python train.py --algo ppo --run
python train.py --algo sac --run
python train.py --algo tqc --run
```

Use a custom checkpoint file:

```bash
python train.py --algo ppo --run --model checkpoints/husky_ppo_ep300.pth
```

Control the number of episodes shown:

```bash
python train.py --algo ppo --run --episodes 10
```

All `--run` flags:

| Flag | Default | Description |
|---|---|---|
| `--run` | (off) | Run inference instead of training |
| `--model` | `checkpoints/husky_<algo>_best.pth` | Path to checkpoint |
| `--episodes` | `5` | Number of episodes to run |

---

### Evaluating and Comparing Algorithms

Run a head-to-head comparison of all three trained algorithms:

```bash
python evaluate.py
```

This requires all three best checkpoints to exist:
- `checkpoints/husky_ppo_best.pth`
- `checkpoints/husky_sac_best.pth`
- `checkpoints/husky_tqc_best.pth`

Options:

| Flag | Default | Description |
|---|---|---|
| `-n`, `--episodes` | `20` | Evaluation episodes per algorithm |
| `-o`, `--output` | `comparison.png` | Output plot filename (saved in `outputs/`) |

Examples:

```bash
python evaluate.py -n 50              # 50 episodes per algorithm
python evaluate.py -n 30 -o out.png   # custom output name
```

**What `evaluate.py` produces:**

1. A **per-episode log** for each algorithm showing reward, steps, final phase, and win/loss
2. A **summary table** printed to the terminal with:
   - Win rate, mean/std/min/max/median episode reward
   - Average steps (all episodes and wins only)
   - Final phase distribution
3. Three **plot files** saved to `outputs/`:
   - `comparison.png` — bar charts, episode reward curves, box plots, phase distributions, action frequencies
   - `comparison_scatter.png` — steps vs reward scatter (wins as stars, losses as crosses)
   - `comparison_cumwins.png` — cumulative wins across evaluation episodes

---

## Checkpoints

Checkpoints are saved to the `checkpoints/` directory:

| File | Saved when |
|---|---|
| `husky_<algo>_best.pth` | New best rolling average reward is achieved |
| `husky_<algo>_ep<N>.pth` | Every 100 episodes |
| `husky_<algo>_final.pth` | End of training (max episodes or early stop) |

---

## Outputs

After running `evaluate.py`, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `comparison.png` | 3×3 grid: reward bars, win rate, steps, episode curves, box plot, phase distribution, action frequency |
| `comparison_scatter.png` | Steps vs episode reward (wins/losses marked) |
| `comparison_cumwins.png` | Cumulative win count over evaluation episodes |

---

## Extending the Project

To add a new RL algorithm:

1. Create `husky_rl/agents/<algo>.py` implementing two functions:

```python
def train(save_prefix: str):
    """Train the agent and save checkpoints to save_prefix_best.pth, etc."""
    ...

def run_trained(model_path: str, n_episodes: int):
    """Load a checkpoint and run inference with GUI enabled."""
    ...
```

2. Add an import case in `train.py` inside `_get_algo()`:

```python
elif name == "your_algo":
    from husky_rl.agents import your_algo
    return your_algo
```

3. Add `"your_algo"` to the `choices` list in the `--algo` argument parser.

The environment (`HuskyTask2Env`) and all shared utilities (`config.py`, `models.py`) are available for import in your agent file.
