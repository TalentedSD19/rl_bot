"""
models.py  --  neural network architectures

ActorCritic        : shared trunk + policy head + value head  (PPO)
QNetwork           : state -> Q-values for all actions          (DQN)
SACDiscreteActor   : state -> action probabilities              (SAC)
SACDiscreteCritic  : state -> Q-values for all actions          (SAC twin)
apply_spin_mask    : action masking shared by all algorithms
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import (
    OBS_DIM, N_ACTIONS, N_PHASES,
    PHASE_FIND_GREEN, PHASE_APPROACH_GREEN,
    PHASE_FIND_RED,   PHASE_APPROACH_RED,
)


# ---------------------------------------------------------------------------
# PPO: Actor-Critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.trunk(x)
        return self.actor(f), self.critic(f).squeeze(-1)

    def get_action(self, s):
        """Sample action, return (action, log_prob, value, entropy)."""
        logits, value = self(s)
        logits = apply_spin_mask(logits, s)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, states, actions):
        """Evaluate stored actions for PPO update."""
        logits, values = self(states)
        logits = apply_spin_mask(logits, states)
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ---------------------------------------------------------------------------
# Shared action masking (PPO + future discrete algos)
# ---------------------------------------------------------------------------

def apply_spin_mask(logits: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """
    Block spinning away from the active target.
    Green obs columns 0,3  for phases FIND_GREEN / APPROACH_GREEN.
    Red   obs columns 4,7  for phases FIND_RED  / APPROACH_RED.
    """
    phase  = (states[:, 9] * (N_PHASES - 1)).round().long()
    masked = logits.clone()

    for is_green in (True, False):
        lo_phase = PHASE_FIND_GREEN     if is_green else PHASE_FIND_RED
        hi_phase = PHASE_APPROACH_GREEN if is_green else PHASE_APPROACH_RED
        cx_col, vis_col = (0, 3) if is_green else (4, 7)

        active = (phase == lo_phase) | (phase == hi_phase)
        vis    = active & (states[:, vis_col] > 0.5)
        if not vis.any():
            continue
        cx           = states[:, cx_col]
        centre_error = (cx - 0.5).abs()

        masked[vis & (cx > 0.53), 0]               = -1e9  # target right  -> block spin-L
        masked[vis & (cx < 0.47), 1]               = -1e9  # target left   -> block spin-R
        masked[vis & (centre_error < 0.12), 0]     = -1e9  # centred       -> no spin
        masked[vis & (centre_error < 0.12), 1]     = -1e9

    return masked


# ---------------------------------------------------------------------------
# DQN: Q-Network
# ---------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al. 2016).

    Shared trunk feeds two separate streams:
      Value stream     V(s)        -- how good is this state in general
      Advantage stream A(s, a)     -- how much better is each action vs average

    Q(s, a) = V(s) + A(s, a) - mean_a'[ A(s, a') ]

    Subtracting the mean advantage makes Q identifiable (otherwise V and A
    can shift by an arbitrary constant and still sum to the same Q).
    This decomposition helps on tasks where some states are clearly
    bad/good regardless of action — e.g. the find / approach phases here.
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.value_stream     = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, n_actions)

    def forward(self, x):
        f = self.trunk(x)
        v = self.value_stream(f)                          # (B, 1)
        a = self.advantage_stream(f)                      # (B, N_ACTIONS)
        return v + (a - a.mean(dim=1, keepdim=True))      # (B, N_ACTIONS)


# ---------------------------------------------------------------------------
# SAC (Discrete): Actor + Critic
# ---------------------------------------------------------------------------

class SACDiscreteActor(nn.Module):
    """
    Outputs a probability distribution over discrete actions.
    Spin mask is applied to logits before softmax so illegal spins
    get zero probability.
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def evaluate(self, states):
        """Returns (probs, log_probs, entropy) — all per-action tensors."""
        logits = self.net(states)
        logits = apply_spin_mask(logits, states)
        probs  = torch.softmax(logits, dim=-1)
        probs_clamped = probs.clamp(min=1e-8)          # numerical safety
        log_probs     = torch.log(probs_clamped)
        entropy       = -(probs * log_probs).sum(1)    # scalar per sample
        return probs, log_probs, entropy


class SACDiscreteCritic(nn.Module):
    """
    Single Q-network: state -> Q-value for every discrete action.
    Instantiate two of these in SACAgent for the twin-critic trick.
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
