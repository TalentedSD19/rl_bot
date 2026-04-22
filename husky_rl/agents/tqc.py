"""
tqc.py  --  Truncated Quantile Critics for Discrete Actions

Adapts Kuznetsov et al. (2020) "Controlling Overestimation Bias with Truncated
Mixture of Continuous Distributional Quantile Critics" to a discrete action
space using the SAC-discrete entropy formulation (Christodoulou 2019).

Key differences from SAC
-------------------------
  - Each critic outputs N_QUANTILES values per action (distributional)
  - N_CRITICS independent critics (default 5, vs SAC's twin)
  - Target quantiles are sorted then the top TOP_DROP_PER_CRITIC per critic
    are discarded before bootstrapping → aggressive anti-overestimation

Update equations
----------------
  Target quantiles (per sample, per kept quantile j):
    z_j = r + γ*(1-d) * Σ_a π(a|s') * [ z_j_trunc(s',a) - α*log π(a|s') ]

  Critic loss (quantile Huber, each critic i):
    L_i = mean_{τ,j}  ρ_τ( z_j - z_i(s,a_taken) )

  Actor loss (same as SAC-discrete):
    L_π = mean( Σ_a π(a|s) * [ α*log π(a|s) - Q_mean(s,a) ] )
    Q_mean = mean over all critics and quantiles

  Alpha loss (same as SAC-discrete):
    L_α = mean( α * [ H(π(s)) - H_target ] )
"""

import copy
import time
import random as pyrandom
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from husky_rl.config import TQC, MAX_STEPS, N_ACTIONS, PHASE_NAMES, ACTION_NAMES
from husky_rl.environment import HuskyTask2Env
from husky_rl.models import SACDiscreteActor, TQCCritic

_C = TQC


# ---------------------------------------------------------------------------
# Replay buffer  (same structure as SAC)
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
# Quantile Huber loss
# ---------------------------------------------------------------------------

def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                         tau: torch.Tensor) -> torch.Tensor:
    """
    pred:   [B, N]   predicted quantile values for the taken action
    target: [B, M]   target quantile values (after truncation)
    tau:    [N]      quantile fractions in (0, 1)

    Returns a scalar loss (mean over batch, pred-quantiles, target-quantiles).
    """
    B, N = pred.shape
    M    = target.shape[1]

    # TD errors: target_j - pred_i  →  [B, N, M]
    td = target.unsqueeze(1) - pred.unsqueeze(2)

    # Huber loss  (smooth_l1 = Huber with δ=1, symmetric)
    huber = F.smooth_l1_loss(
        pred.unsqueeze(2).expand(B, N, M),
        target.unsqueeze(1).expand(B, N, M),
        reduction='none',
    )

    # Asymmetric quantile weighting
    tau_w = tau.view(1, N, 1)                          # [1, N, 1]
    loss  = (tau_w - (td.detach() < 0).float()).abs() * huber
    return loss.mean()


# ---------------------------------------------------------------------------
# TQC Agent
# ---------------------------------------------------------------------------

class TQCAgent:
    def __init__(self, device):
        self.device = device

        n_q  = _C["N_QUANTILES"]
        n_c  = _C["N_CRITICS"]
        drop = _C["TOP_DROP_PER_CRITIC"]
        h    = _C["HIDDEN"]

        self.n_quantiles = n_q
        self.n_critics   = n_c
        # Number of quantile atoms kept per sample after truncation
        self.n_target_q  = n_c * (n_q - drop)

        # Actor (identical architecture to SAC-discrete)
        self.actor = SACDiscreteActor(hidden=h).to(device)

        # N independent critics + frozen target copies
        self.critics = nn.ModuleList(
            [TQCCritic(n_quantiles=n_q, hidden=h).to(device) for _ in range(n_c)]
        )
        self.critics_target = copy.deepcopy(self.critics)
        for p in self.critics_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=_C["LR_ACTOR"])
        self.opt_critic = optim.Adam(
            [p for c in self.critics for p in c.parameters()],
            lr=_C["LR_CRITIC"],
        )

        # Learnable temperature (same target as SAC)
        self.target_entropy = 0.98 * np.log(N_ACTIONS)
        self.log_alpha = torch.tensor(
            np.log(_C["ALPHA_INIT"]), dtype=torch.float32,
            requires_grad=True, device=device,
        )
        self.opt_alpha = optim.Adam([self.log_alpha], lr=_C["LR_ALPHA"])

        # Fixed quantile midpoints  τ_i = (2i+1)/(2N)
        self.tau = torch.FloatTensor(
            [(2 * i + 1) / (2 * n_q) for i in range(n_q)]
        ).to(device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray) -> int:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _, _ = self.actor.evaluate(s_t)
        return int(probs.argmax(1).item())

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        B     = states.shape[0]
        alpha = self.alpha

        # -- Critic targets --------------------------------------------------
        with torch.no_grad():
            probs_next, lp_next, _ = self.actor.evaluate(next_states)

            # Stack target critics: [B, N_C, N_A, N_Q]
            tgt_all = torch.stack([c(next_states) for c in self.critics_target], dim=1)
            # Rearrange to [B, N_A, N_C, N_Q] then flatten last two → [B, N_A, N_C*N_Q]
            tgt_all = tgt_all.permute(0, 2, 1, 3).reshape(B, N_ACTIONS, -1)

            # Sort per action and drop top atoms (keep lowest N_C*(N_Q-drop))
            tgt_sorted, _ = tgt_all.sort(dim=-1)
            tgt_trunc = tgt_sorted[:, :, : self.n_target_q]        # [B, N_A, M]

            # Entropy correction: subtract α*log π(a|s') per action
            tgt_trunc = tgt_trunc - alpha.detach() * lp_next.unsqueeze(2)

            # Expectation over actions → [B, M]
            target_q = (probs_next.unsqueeze(2) * tgt_trunc).sum(dim=1)
            target_q = rewards + _C["GAMMA"] * (1.0 - dones) * target_q

        # -- Critic update ---------------------------------------------------
        # Each critic: select taken action quantiles [B, N_Q], compare to [B, M]
        pred_qs     = [c(states)[range(B), actions] for c in self.critics]
        critic_loss = sum(
            quantile_huber_loss(pq, target_q, self.tau) for pq in pred_qs
        )

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for c in self.critics for p in c.parameters()], _C["GRAD_CLIP"]
        )
        self.opt_critic.step()

        # -- Actor update ----------------------------------------------------
        probs, log_probs, entropy = self.actor.evaluate(states)
        # Mean Q over all critics and all quantiles → [B, N_A]
        with torch.no_grad():
            q_all  = torch.stack([c(states) for c in self.critics], dim=1)  # [B, N_C, N_A, N_Q]
            q_mean = q_all.mean(dim=1).mean(dim=-1)                          # [B, N_A]

        actor_loss = (probs * (alpha.detach() * log_probs - q_mean)).sum(dim=1).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), _C["GRAD_CLIP"])
        self.opt_actor.step()

        # -- Alpha update ----------------------------------------------------
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha.clamp_(_C["LOG_ALPHA_MIN"], _C["LOG_ALPHA_MAX"])

        # -- Soft target update (Polyak) -------------------------------------
        tau = _C["TAU"]
        for c, ct in zip(self.critics, self.critics_target):
            for p, pt in zip(c.parameters(), ct.parameters()):
                pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

        return (
            critic_loss.item() / self.n_critics,
            actor_loss.item(),
            self.alpha.item(),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(save_prefix: str = "checkpoints/husky_tqc"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env   = HuskyTask2Env(gui=False)
    agent = TQCAgent(device)
    buf   = ReplayBuffer(_C["REPLAY_SIZE"])

    ep_rewards  = []
    wins        = []
    best_avg    = -float("inf")
    total_steps = 0

    print(f"\nStarting TQC  n_critics={_C['N_CRITICS']}  "
          f"n_quantiles={_C['N_QUANTILES']}  "
          f"top_drop_per_critic={_C['TOP_DROP_PER_CRITIC']}  "
          f"batch={_C['BATCH_SIZE']}")
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
            buf.push(state, action, reward, next_state, done)

            state        = next_state
            ep_r        += reward
            total_steps += 1

            if len(buf) >= _C["BATCH_SIZE"]:
                cl, al, _ = agent.update(buf.sample(_C["BATCH_SIZE"]))
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
            torch.save({
                "actor":     agent.actor.state_dict(),
                "critics":   [c.state_dict() for c in agent.critics],
                "log_alpha": agent.log_alpha.item(),
            }, f"{save_prefix}_best.pth")

        tag = "WIN" if ep_won else "   "
        print(f"{ep:4d} {ep_r:8.2f} {avg:8.2f} "
              f"{win_rate*100:6.1f} {step+1:5d} {phase_tag:>7} "
              f"{alpha_val:6.3f} {best_avg:8.2f}  {tag}")

        if ep % 100 == 0:
            ckpt   = f"{save_prefix}_ep{ep}.pth"
            avg_cl = cl_sum / n_updates if n_updates else 0.0
            avg_al = al_sum / n_updates if n_updates else 0.0
            torch.save({
                "actor":     agent.actor.state_dict(),
                "critics":   [c.state_dict() for c in agent.critics],
                "log_alpha": agent.log_alpha.item(),
            }, ckpt)
            print(f"  ↳ checkpoint {ckpt}  "
                  f"[C {avg_cl:.4f}  A {avg_al:.4f}  "
                  f"alpha {alpha_val:.3f}  steps {total_steps}]")

        if len(wins) >= 50 and win_rate >= _C["EARLY_STOP_RATE"]:
            print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
            env.close()
            torch.save({
                "actor":     agent.actor.state_dict(),
                "critics":   [c.state_dict() for c in agent.critics],
                "log_alpha": agent.log_alpha.item(),
            }, f"{save_prefix}_final.pth")
            print(f"Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
            return agent

    env.close()
    torch.save({
        "actor":     agent.actor.state_dict(),
        "critics":   [c.state_dict() for c in agent.critics],
        "log_alpha": agent.log_alpha.item(),
    }, f"{save_prefix}_final.pth")
    print(f"Done. Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
    return agent


# ---------------------------------------------------------------------------
# Inference / visualisation
# ---------------------------------------------------------------------------

def run_trained(model_path: str = "checkpoints/husky_tqc_best.pth", n_episodes: int = 5):
    device = torch.device("cpu")
    agent  = TQCAgent(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    for i, c in enumerate(agent.critics):
        c.load_state_dict(ckpt["critics"][i])
    agent.log_alpha.data.fill_(ckpt["log_alpha"])
    agent.actor.eval()
    print(f"Loaded {model_path}")

    env  = HuskyTask2Env(gui=True)
    wins = 0

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0

        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, _, _ = agent.actor.evaluate(s_t)
                action      = probs.argmax(1).item()

            state, reward, done = env.step(action)
            total_r += reward

            print(f"ep {ep:2d} step {step:3d} | {PHASE_NAMES[env.phase]} "
                  f"| {ACTION_NAMES[action]} "
                  f"| g_vis={int(bool(state[3]))} g_area={state[2]:.3f} "
                  f"| r_vis={int(bool(state[7]))} r_area={state[6]:.3f} "
                  f"| r={reward:+.2f}")

            if done:
                wins += 1
                print(f"  *** WIN!  ep_reward={total_r:.1f} ***\n")
                time.sleep(2)
                break
        else:
            print(f"  ep {ep} timeout  ep_reward={total_r:.1f}\n")

    env.close()
    print(f"\nWin rate: {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")
