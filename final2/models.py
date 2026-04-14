"""
models.py  --  neural network architectures

ActorCritic   : shared trunk + policy head + value head  (used by PPO)
apply_spin_mask : action masking based on camera observation

When adding DQN: put QNetwork here.
When adding SAC: put SACActorNetwork + SACCriticNetwork here.
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
# Placeholder: DQN Q-Network (add implementation when ready)
# ---------------------------------------------------------------------------
# class QNetwork(nn.Module):
#     def __init__(self, state_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden),   nn.ReLU(),
#             nn.Linear(hidden, n_actions),
#         )
#     def forward(self, x):
#         return self.net(x)


# ---------------------------------------------------------------------------
# Placeholder: SAC Actor / Twin Critics (add implementation when ready)
# ---------------------------------------------------------------------------
# class SACActorNetwork(nn.Module): ...
# class SACCriticNetwork(nn.Module): ...
