"""
train.py  --  entry point

Usage:
  python train.py                         # train with PPO (default)
  python train.py --algo ppo              # train with PPO
  python train.py --algo sac              # train with SAC
  python train.py --algo tqc              # train with TQC
  python train.py --algo ppo --run        # watch best PPO checkpoint
  python train.py --algo ppo --run --model checkpoints/husky_ppo_best.pth

Adding a new algorithm:
  1. Create husky_rl/agents/<algo>.py with train(save_prefix) / run_trained(model_path, n_episodes)
  2. Import it below and add a case in _get_algo().
"""

import argparse
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _get_algo(name: str):
    if name == "ppo":
        from husky_rl.agents import ppo
        return ppo
    elif name == "sac":
        from husky_rl.agents import sac
        return sac
    elif name == "tqc":
        from husky_rl.agents import tqc
        return tqc
    raise ValueError(f"Unknown algorithm: {name!r}. Choose from: ppo, sac, tqc")


def main():
    parser = argparse.ArgumentParser(description="Husky RL trainer")
    parser.add_argument("--algo",   default="ppo",
                        choices=["ppo", "sac", "tqc"],
                        help="RL algorithm to use")
    parser.add_argument("--run",    action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--model",  default=None,
                        help="Checkpoint path for --run (default: checkpoints/husky_<algo>_best.pth)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes for --run (default: 5)")
    args = parser.parse_args()

    algo        = _get_algo(args.algo)
    save_prefix = f"checkpoints/husky_{args.algo}"

    if args.run:
        model_path = args.model or f"{save_prefix}_best.pth"
        algo.run_trained(model_path=model_path, n_episodes=args.episodes)
    else:
        algo.train(save_prefix=save_prefix)


if __name__ == "__main__":
    main()
