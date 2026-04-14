"""
dqn.py  --  Deep Q-Network

  ReplayBuffer  : uniform experience replay
  dqn_update    : one gradient step on a sampled mini-batch
  train         : full training loop
  run_trained   : load checkpoint and visualise
"""

import time
import random as pyrandom
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DQN, MAX_STEPS, PHASE_NAMES, ACTION_NAMES
from environment import HuskyTask2Env
from models import QNetwork, apply_spin_mask

_C = DQN


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = pyrandom.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# DQN update
# ---------------------------------------------------------------------------

def dqn_update(policy, target, optimizer, batch, device):
    states, actions, rewards, next_states, dones = batch

    states      = torch.FloatTensor(states).to(device)
    actions     = torch.LongTensor(actions).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones       = torch.FloatTensor(dones).to(device)

    # Current Q-values for taken actions
    q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: greedy next Q with spin mask applied
    with torch.no_grad():
        next_q     = apply_spin_mask(target(next_states), next_states)
        max_next_q = next_q.max(1)[0]
        target_q   = rewards + _C["GAMMA"] * (1.0 - dones) * max_next_q

    loss = nn.functional.smooth_l1_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(save_prefix: str = "husky_dqn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env    = HuskyTask2Env(gui=False)
    policy = QNetwork(hidden=_C["HIDDEN"]).to(device)
    target = QNetwork(hidden=_C["HIDDEN"]).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=_C["LR"])
    buffer    = ReplayBuffer(_C["REPLAY_SIZE"])

    eps         = _C["EPS_START"]
    ep_rewards  = []
    wins        = []
    best_avg    = -float("inf")
    total_steps = 0

    print(f"\nStarting DQN  eps_start={eps}  eps_end={_C['EPS_END']}  "
          f"decay={_C['EPS_DECAY']}  target_update={_C['TARGET_UPDATE']}")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'Phase':>7} {'Eps':>6} {'BestAvg':>8}  Result")

    for ep in range(1, _C["MAX_EPISODES"] + 1):
        state     = env.reset()
        ep_r      = 0.0
        ep_won    = False
        loss_sum  = 0.0
        n_updates = 0

        for step in range(MAX_STEPS):
            # Epsilon-greedy with spin mask on the greedy branch
            if pyrandom.random() < eps:
                action = pyrandom.randint(0, 4)
            else:
                with torch.no_grad():
                    s_t    = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q      = apply_spin_mask(policy(s_t), s_t)
                    action = q.argmax(1).item()

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state        = next_state
            ep_r        += reward
            total_steps += 1

            if len(buffer) >= _C["BATCH_SIZE"]:
                loss      = dqn_update(policy, target, optimizer,
                                       buffer.sample(_C["BATCH_SIZE"]), device)
                loss_sum  += loss
                n_updates += 1

            if total_steps % _C["TARGET_UPDATE"] == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                ep_won = True
                break

        eps = max(_C["EPS_END"], eps * _C["EPS_DECAY"])

        ep_rewards.append(ep_r)
        wins.append(int(ep_won))
        avg       = np.mean(ep_rewards[-20:])
        win_rate  = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
        phase_tag = PHASE_NAMES[env.phase]

        if avg > best_avg:
            best_avg = avg
            torch.save(policy.state_dict(), f"{save_prefix}_best.pth")

        tag = "WIN" if ep_won else "   "
        print(f"{ep:4d} {ep_r:8.2f} {avg:8.2f} "
              f"{win_rate*100:6.1f} {step+1:5d} {phase_tag:>7} "
              f"{eps:6.3f} {best_avg:8.2f}  {tag}")

        if ep % 100 == 0:
            ckpt     = f"{save_prefix}_ep{ep}.pth"
            avg_loss = loss_sum / n_updates if n_updates else 0.0
            torch.save(policy.state_dict(), ckpt)
            print(f"  ↳ checkpoint {ckpt}  "
                  f"[loss {avg_loss:.4f}  steps {total_steps}  eps {eps:.3f}]")

        if len(wins) >= 50 and win_rate >= _C["EARLY_STOP_RATE"]:
            print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
            env.close()
            torch.save(policy.state_dict(), f"{save_prefix}_final.pth")
            print(f"Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
            return policy

    env.close()
    torch.save(policy.state_dict(), f"{save_prefix}_final.pth")
    print(f"Done. Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
    return policy


# ---------------------------------------------------------------------------
# Inference / visualisation
# ---------------------------------------------------------------------------

def run_trained(model_path: str = "husky_dqn_best.pth", n_episodes: int = 5):
    device = torch.device("cpu")
    policy = QNetwork(hidden=_C["HIDDEN"]).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    policy.eval()
    print(f"Loaded {model_path}")

    env  = HuskyTask2Env(gui=True)
    wins = 0

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0

        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q      = apply_spin_mask(policy(s_t), s_t)
                action = q.argmax(1).item()

            state, reward, done = env.step(action)
            total_r += reward

            print(f"ep {ep:2d} step {step:3d} | {PHASE_NAMES[env.phase]} "
                  f"| {ACTION_NAMES[action]} "
                  f"| g_vis={int(bool(state[3]))} g_area={state[2]:.3f} "
                  f"| r_vis={int(bool(state[7]))} r_area={state[6]:.3f} "
                  f"| r={reward:+.2f}")

            time.sleep(1 / 60)

            if done:
                wins += 1
                print(f"  *** WIN!  ep_reward={total_r:.1f} ***\n")
                time.sleep(2)
                break
        else:
            print(f"  ep {ep} timeout  ep_reward={total_r:.1f}\n")

    env.close()
    print(f"\nWin rate: {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")
