"""
main.py  --  entry point

Usage:
  python main.py                         # train with PPO (default)
  python main.py --algo ppo              # train with PPO
  python main.py --algo ppo --run        # watch best PPO checkpoint
  python main.py --algo ppo --run --model path/to/checkpoint.pth

Adding a new algorithm:
  1. Create dqn.py  with  train(save_prefix) / run_trained(model_path, n_episodes)
  2. Create sac.py  with  train(save_prefix) / run_trained(model_path, n_episodes)
  3. Import them below and add a case in _get_algo().
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
        import ppo
        return ppo
    elif name == "dqn":
        import dqn
        return dqn
    elif name == "sac":
        import sac
        return sac
    raise ValueError(f"Unknown algorithm: {name!r}. Choose from: ppo, dqn, sac")


def main():
    parser = argparse.ArgumentParser(description="Husky RL trainer")
    parser.add_argument("--algo",   default="ppo",
                        choices=["ppo", "dqn", "sac"],
                        help="RL algorithm to use")
    parser.add_argument("--run",    action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--model",  default=None,
                        help="Checkpoint path for --run (default: <algo>_best.pth)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes for --run (default: 5)")
    args = parser.parse_args()

    algo        = _get_algo(args.algo)
    save_prefix = f"husky_{args.algo}"

    if args.run:
        model_path = args.model or f"{save_prefix}_best.pth"
        algo.run_trained(model_path=model_path, n_episodes=args.episodes)
    else:
        algo.train(save_prefix=save_prefix)


if __name__ == "__main__":
    main()
