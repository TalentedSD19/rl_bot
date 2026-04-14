"""
sac.py  --  Soft Actor-Critic for Discrete Actions
            (Christodoulou 2019)

  ReplayBuffer  : uniform experience replay (same shape as DQN's)
  SACAgent      : actor + twin critics + target critics + auto-alpha
  train         : full training loop
  run_trained   : load actor checkpoint and visualise

Update equations
----------------
  Soft value of next state:
    V(s') = sum_a  pi(a|s') * [ Q_min(s',a) - alpha * log pi(a|s') ]

  Critic targets:
    y = r + gamma * (1-done) * V(s')

  Actor loss (minimise):
    L_pi = mean( sum_a  pi(a|s) * [ alpha * log pi(a|s) - Q_min(s,a) ] )

  Alpha (temperature) loss:
    L_alpha = mean( alpha * [ H(pi(s)) - target_entropy ] )
    target_entropy = log(N_ACTIONS)   (maximum entropy for discrete)
"""

import time
import random as pyrandom
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import SAC, MAX_STEPS, N_ACTIONS, PHASE_NAMES, ACTION_NAMES
from environment import HuskyTask2Env
from models import SACDiscreteActor, SACDiscreteCritic

_C = SAC


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
# SAC agent
# ---------------------------------------------------------------------------

class SACAgent:
    def __init__(self, device):
        self.device = device

        self.actor   = SACDiscreteActor(hidden=_C["HIDDEN"]).to(device)
        self.critic1 = SACDiscreteCritic(hidden=_C["HIDDEN"]).to(device)
        self.critic2 = SACDiscreteCritic(hidden=_C["HIDDEN"]).to(device)
        self.target1 = SACDiscreteCritic(hidden=_C["HIDDEN"]).to(device)
        self.target2 = SACDiscreteCritic(hidden=_C["HIDDEN"]).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.eval()
        self.target2.eval()

        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=_C["LR_ACTOR"])
        self.opt_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=_C["LR_CRITIC"])

        # Learnable temperature.
        # Target is 98% of theoretical max entropy (log N_ACTIONS for uniform
        # policy over N_ACTIONS).  Using 100% means the policy must be
        # perfectly uniform to satisfy the constraint — in practice entropy
        # is always below that, so alpha never stops growing.
        self.target_entropy = 0.98 * np.log(N_ACTIONS)
        self.log_alpha      = torch.tensor(
            np.log(_C["ALPHA_INIT"]), dtype=torch.float32,
            requires_grad=True, device=device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=_C["LR_ALPHA"])

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray) -> int:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _, _ = self.actor.evaluate(s_t)
        return probs.argmax(1).item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        alpha = self.alpha.detach()

        # -- Critic update ---------------------------------------------------
        with torch.no_grad():
            next_probs, next_log_probs, _ = self.actor.evaluate(next_states)
            q1_next  = self.target1(next_states)
            q2_next  = self.target2(next_states)
            q_min    = torch.min(q1_next, q2_next)
            # Soft state-value: expectation over all actions
            v_next   = (next_probs * (q_min - alpha * next_log_probs)).sum(1)
            target_q = rewards + _C["GAMMA"] * (1.0 - dones) * v_next

        q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=_C["GRAD_CLIP"])
        self.opt_critic.step()

        # -- Actor update ----------------------------------------------------
        probs, log_probs, entropy = self.actor.evaluate(states)
        q1_det = self.critic1(states).detach()
        q2_det = self.critic2(states).detach()
        q_min  = torch.min(q1_det, q2_det)
        # Minimise: E[ pi * (alpha * log_pi - Q_min) ]
        actor_loss = (probs * (alpha * log_probs - q_min)).sum(1).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # -- Alpha update ----------------------------------------------------
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        # Hard clamp: keep log_alpha in [LOG_ALPHA_MIN, LOG_ALPHA_MAX] so
        # alpha never exceeds ~e^2 ≈ 7.4 regardless of entropy deficit.
        with torch.no_grad():
            self.log_alpha.clamp_(_C["LOG_ALPHA_MIN"], _C["LOG_ALPHA_MAX"])

        # -- Soft target update (Polyak) -------------------------------------
        tau = _C["TAU"]
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        return (critic_loss.item(), actor_loss.item(),
                alpha_loss.item(), self.alpha.item())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(save_prefix: str = "husky_sac"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env    = HuskyTask2Env(gui=False)
    agent  = SACAgent(device)
    buffer = ReplayBuffer(_C["REPLAY_SIZE"])

    ep_rewards  = []
    wins        = []
    best_avg    = -float("inf")
    total_steps = 0

    print(f"\nStarting SAC (discrete)  batch={_C['BATCH_SIZE']}  "
          f"alpha_init={_C['ALPHA_INIT']}  tau={_C['TAU']}")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'Phase':>7} {'Alpha':>6} {'BestAvg':>8}  Result")

    for ep in range(1, _C["MAX_EPISODES"] + 1):
        state     = env.reset()
        ep_r      = 0.0
        ep_won    = False
        cl_sum    = 0.0
        al_sum    = 0.0
        n_updates = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state        = next_state
            ep_r        += reward
            total_steps += 1

            if len(buffer) >= _C["BATCH_SIZE"]:
                cl, al, _, _ = agent.update(buffer.sample(_C["BATCH_SIZE"]))
                cl_sum    += cl
                al_sum    += al
                n_updates += 1

            if done:
                ep_won = True
                break

        ep_rewards.append(ep_r)
        wins.append(int(ep_won))
        avg       = np.mean(ep_rewards[-20:])
        win_rate  = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
        phase_tag = PHASE_NAMES[env.phase]
        alpha_val = agent.alpha.item()

        if avg > best_avg:
            best_avg = avg
            torch.save(agent.actor.state_dict(), f"{save_prefix}_best.pth")

        tag = "WIN" if ep_won else "   "
        print(f"{ep:4d} {ep_r:8.2f} {avg:8.2f} "
              f"{win_rate*100:6.1f} {step+1:5d} {phase_tag:>7} "
              f"{alpha_val:6.3f} {best_avg:8.2f}  {tag}")

        if ep % 100 == 0:
            ckpt   = f"{save_prefix}_ep{ep}.pth"
            avg_cl = cl_sum / n_updates if n_updates else 0.0
            avg_al = al_sum / n_updates if n_updates else 0.0
            torch.save(agent.actor.state_dict(), ckpt)
            print(f"  ↳ checkpoint {ckpt}  "
                  f"[C {avg_cl:.4f}  A {avg_al:.4f}  "
                  f"alpha {alpha_val:.3f}  steps {total_steps}]")

        if len(wins) >= 50 and win_rate >= _C["EARLY_STOP_RATE"]:
            print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
            env.close()
            torch.save(agent.actor.state_dict(), f"{save_prefix}_final.pth")
            print(f"Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
            return agent

    env.close()
    torch.save(agent.actor.state_dict(), f"{save_prefix}_final.pth")
    print(f"Done. Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
    return agent


# ---------------------------------------------------------------------------
# Inference / visualisation
# ---------------------------------------------------------------------------

def run_trained(model_path: str = "husky_sac_best.pth", n_episodes: int = 5):
    device = torch.device("cpu")
    actor  = SACDiscreteActor(hidden=_C["HIDDEN"]).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    actor.eval()
    print(f"Loaded {model_path}")

    env  = HuskyTask2Env(gui=True)
    wins = 0

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0

        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, _, _ = actor.evaluate(s_t)
                action      = probs.argmax(1).item()

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
