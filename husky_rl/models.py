"""
models.py  --  neural network architectures

ActorCritic        : shared trunk + policy head + value head  (PPO)
SACDiscreteActor   : state -> action probabilities             (SAC)
SACDiscreteCritic  : state -> Q-values for all actions         (SAC twin)
apply_spin_mask    : action masking shared by all algorithms
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

from husky_rl.config import (
    OBS_DIM, N_ACTIONS, N_PHASES,
    PHASE_FIND_GREEN, PHASE_APPROACH_GREEN,
    PHASE_FIND_RED,   PHASE_APPROACH_RED,
)


# ---------------------------------------------------------------------------
# PPO: Actor-Critic
# ---------------------------------------------------------------------------

def _ortho(in_f: int, out_f: int, gain: float = math.sqrt(2)) -> nn.Linear:
    """Linear layer with orthogonal init — standard PPO best practice."""
    layer = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_f: int, hidden: int) -> nn.Sequential:
    """3-layer ReLU trunk with orthogonal init."""
    return nn.Sequential(
        _ortho(in_f,  hidden), nn.ReLU(),
        _ortho(hidden, hidden), nn.ReLU(),
        _ortho(hidden, hidden), nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """
    Separate actor / critic networks — avoids gradient interference.
    3-layer ReLU trunk (256 units) with orthogonal initialisation.
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.actor_net  = _mlp(state_dim, hidden)
        self.critic_net = _mlp(state_dim, hidden)
        self.actor_head  = _ortho(hidden, n_actions, gain=0.01)
        self.critic_head = _ortho(hidden, 1,         gain=1.0)

    def forward(self, x):
        logits = self.actor_head(self.actor_net(x))
        value  = self.critic_head(self.critic_net(x)).squeeze(-1)
        return logits, value

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


# ---------------------------------------------------------------------------
# TQC: Distributional Critic
# ---------------------------------------------------------------------------

class TQCCritic(nn.Module):
    """
    State -> N_ACTIONS × N_QUANTILES quantile values.
    Instantiate N_CRITICS of these in TQCAgent.
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 n_quantiles: int = 25, hidden: int = 256):
        super().__init__()
        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, N_ACTIONS, N_QUANTILES]."""
        return self.net(x).view(x.shape[0], self.n_actions, self.n_quantiles)
