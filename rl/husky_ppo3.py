"""
Husky PPO 3 — Camera-only red-cylinder chasing
===============================================

Behaviour rules (hard-coded into rewards):
  1. If cylinder NOT visible  → spin to search (forward/approach is penalised)
  2. The moment cylinder IS visible → stop spinning immediately (spinning penalised)
  3. Keep the cylinder horizontally centred
  4. Approach the cylinder only while it is visible and reasonably centred
  5. Win when the cylinder fills > 50% of the camera frame
  6. Cylinder is a static body (mass=0) so it won't glitch when the camera touches it

Actions (5 discrete):
  0  spin left          (-spd, +spd)
  1  spin right         (+spd, -spd)
  2  forward            (+spd, +spd)
  3  forward + steer L  (+spd*0.4, +spd)
  4  forward + steer R  (+spd, +spd*0.4)

State (4 floats):
  [cx, cy, area, visible]   — all zero when cylinder not detected

Usage:
  python husky_ppo3.py          # train
  python husky_ppo3.py run      # run best checkpoint in GUI
"""

import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import pybullet as p
import pybullet_data

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Simulation ──────────────────────────────────────────────────────────────
SPEED        = 15
LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]

CAM_W, CAM_H = 320, 240
FOV          = 90

# Red pixel detection thresholds
RED_R_MIN = 150
RED_G_MAX = 80
RED_B_MAX = 80
MIN_RED_PIXELS = 5          # ignore stray pixels

# Action indices for spin actions
SPIN_ACTIONS = [0, 1]

CYL_RADIUS = 0.7
CYL_HEIGHT = 1.0

# ─── PPO hyper-parameters ────────────────────────────────────────────────────
MAX_EPISODES    = 500
MAX_STEPS       = 300       # steps per episode before timeout
ROLLOUT_STEPS   = 2048      # env steps to collect before each PPO update
N_EPOCHS        = 10        # gradient epochs per rollout
MINI_BATCH      = 64
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.05
VALUE_COEF      = 0.5
LR              = 3e-4
GRAD_CLIP       = 0.5
WIN_AREA        = 0.50      # fraction of frame that counts as "reached"
EARLY_STOP_RATE = 0.80      # stop training if win-rate over last 50 eps > 80 %


# ═══════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

class HuskyEnv:
    """PyBullet Husky environment — camera is the only sensor."""

    def __init__(self, gui=False):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        self._build_world()

    # ── World setup ──────────────────────────────────────────────────────────

    def _build_world(self):
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1],
                                physicsClientId=self.client)
        self._build_cylinder()

    def _build_cylinder(self):
        """Static red cylinder (mass=0 → never moves, never glitches on touch)."""
        col = p.createCollisionShape(p.GEOM_CYLINDER,
                                     radius=CYL_RADIUS, height=CYL_HEIGHT,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=CYL_RADIUS, length=CYL_HEIGHT,
                                  rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                  physicsClientId=self.client)
        self.cyl_id = p.createMultiBody(
            baseMass=0,                         # static — won't be pushed
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[3, 0, CYL_HEIGHT / 2],
            physicsClientId=self.client,
        )

    def _place_cylinder_random(self):
        """Teleport cylinder to a random position around the robot."""
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(3.0, 6.0)
        pos   = [dist * np.cos(angle), dist * np.sin(angle), CYL_HEIGHT / 2]
        p.resetBasePositionAndOrientation(self.cyl_id, pos, [0, 0, 0, 1],
                                          physicsClientId=self.client)

    # ── Actuation ────────────────────────────────────────────────────────────

    def _drive(self, left_vel, right_vel):
        for j in LEFT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=left_vel, force=100,
                                    physicsClientId=self.client)
        for j in RIGHT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=right_vel, force=100,
                                    physicsClientId=self.client)

    # ── Camera ───────────────────────────────────────────────────────────────

    def _get_rgb(self):
        pos, ori = p.getBasePositionAndOrientation(self.husky,
                                                   physicsClientId=self.client)
        rot    = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        eye    = np.array(pos) + rot @ [0.5, 0.0, 0.3]
        target = eye + rot @ [1.0, 0.0, 0.0]
        up     = rot @ [0.0, 0.0, 1.0]
        view   = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj   = p.computeProjectionMatrixFOV(FOV, CAM_W / CAM_H, 0.02, 20.0)
        _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=self.client)
        return np.array(rgb, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]

    @staticmethod
    def detect_cylinder(rgb):
        """
        Finds red pixels in the image.
        Returns (cx, cy, area) if found, else None.
          cx   — horizontal centre in [0, 1],  0.5 = perfectly centred
          cy   — vertical centre in [0, 1]
          area — fraction of total pixels that are red
        """
        mask = (
            (rgb[:, :, 0] > RED_R_MIN) &
            (rgb[:, :, 1] < RED_G_MAX) &
            (rgb[:, :, 2] < RED_B_MAX)
        )
        n = int(mask.sum())
        if n < MIN_RED_PIXELS:
            return None
        ys, xs = np.where(mask)
        cx   = xs.mean() / CAM_W          # 0 = left edge, 1 = right edge
        cy   = ys.mean() / CAM_H
        area = n / (CAM_W * CAM_H)
        return cx, cy, area

    # ── Gym-style interface ──────────────────────────────────────────────────

    def reset(self):
        # Reset robot pose and velocity
        p.resetBasePositionAndOrientation(self.husky, [0, 0, 0.1], [0, 0, 0, 1],
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.husky, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.client)
        for j in range(p.getNumJoints(self.husky, physicsClientId=self.client)):
            p.resetJointState(self.husky, j, 0, 0, physicsClientId=self.client)
        self._drive(0, 0)

        self._place_cylinder_random()

        # Let the sim settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        self.step_count          = 0
        self.prev_visible        = False
        self.prev_area           = 0.0
        self.prev_centre_error   = 0.5     # worst-case starting assumption
        self.discovery_rewarded  = False
        return self._get_obs()

    def _get_obs(self):
        """
        Observation: [cx, cy, area, visible]
        Everything is 0 when the cylinder is not in sight.
        """
        info = self.detect_cylinder(self._get_rgb())
        if info is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cx, cy, area = info
        return np.array([cx, cy, area, 1.0], dtype=np.float32)

    def step(self, action):
        # Map action index → wheel velocities
        vel_map = {
            0: (-SPEED,          +SPEED),       # spin left
            1: (+SPEED,          -SPEED),       # spin right
            2: (+SPEED,          +SPEED),       # forward
            3: (+SPEED * 0.4,    +SPEED),       # forward + steer left
            4: (+SPEED,          +SPEED * 0.4), # forward + steer right
        }
        self._drive(*vel_map[action])

        for _ in range(4):                      # 4 sim sub-steps per env step
            p.stepSimulation(physicsClientId=self.client)

        self.step_count += 1
        obs            = self._get_obs()
        reward, done   = self._reward(obs, action)

        # Remember for next step
        self.prev_visible = bool(obs[3])
        if obs[3]:
            self.prev_area         = obs[2]
            self.prev_centre_error = abs(obs[0] - 0.5)

        return obs, reward, done

    def _reward(self, obs, action):
        """
        Reward function — changes from original:

          1. Discovery bonus is one-time only (self.discovery_rewarded guard).
             Prevents the oscillation exploit of spinning in/out of frame for +3
             every cycle.

          2. Dropping the cylinder from view now gives a hard -5.0 penalty.
             This makes oscillation immediately self-defeating regardless of
             discovery bonus.

          3. Win condition returns a clean +100.0 with no partial-step noise
             (old code returned r + 100 which could add/subtract up to ~3).

          4. Time pressure halved (−0.1 → −0.05) to give the agent more room
             to plan without panic-rushing while off-centre.
        """
        cx, cy, area, visible = obs
        visible = bool(visible)
        r = 0.0

        # ── Cylinder NOT visible ─────────────────────────────────────────────
        if not visible:
            # Penalty for losing the cylinder — closes the oscillation exploit.
            # The agent had it, now it doesn't: that's bad.
            if self.prev_visible:
                r -= 5.0

            # Blind movement penalties
            if action == 2:             # driving forward with no target
                r -= 1.0
            elif action in (3, 4):      # steer-forward blind
                r -= 0.5

            # Light time pressure — encourages fast search without panic
            r -= 0.05
            return r, False

        # ── Cylinder IS visible ──────────────────────────────────────────────

        # One-time discovery bonus — cannot be farmed by oscillating in/out.
        # self.discovery_rewarded is reset to False in reset().
        if not self.prev_visible and not self.discovery_rewarded:
            r += 3.0
            self.discovery_rewarded = True

        # Win condition — clean signal, no partial-step noise added
        if area > WIN_AREA:
            return 100.0, True

        centre_error = abs(cx - 0.5)    # 0 = perfect centre, 0.5 = screen edge

        # --- Centering: reward for REDUCING centre_error since last step ---
        # This is the key signal — tells the agent "you're getting closer to centre"
        delta_ce = self.prev_centre_error - centre_error   # positive = improving
        r += delta_ce * 5.0

        # Small static penalty so staying off-centre is never free
        r -= centre_error * 0.5

        # --- Directional steering reward ---
        # Explicitly reward the action that moves the cylinder toward centre.
        # cx > 0.5 → cylinder is on the RIGHT → turn right (actions 1 or 4)
        # cx < 0.5 → cylinder is on the LEFT  → turn left  (actions 0 or 3)
        if cx > 0.53:                           # cylinder on right
            if action in (1, 4):               # spin-right or fwd+steer-right
                r += 0.4
        elif cx < 0.47:                         # cylinder on left
            if action in (0, 3):               # spin-left or fwd+steer-left
                r += 0.4

        # --- Approach reward: only when reasonably centred ---
        # Tighter: only approach when well centred
        if centre_error < 0.12:          # was 0.25 — cx must be 0.38–0.62
            delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
            r += delta_area * 30.0
        elif centre_error < 0.20:        # partially centred — approach weakly rewarded
            delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
            r += delta_area * 10.0
        else:
            if action in (2, 3, 4):      # penalise ALL forward actions when off-centre
                r -= 0.4                 # was only action==2

        # Light time pressure
        r -= 0.05
        return r, False

    def close(self):
        p.disconnect(self.client)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ACTOR-CRITIC NETWORK
# ═══════════════════════════════════════════════════════════════════════════

def apply_spin_mask(logits, state_tensor):
    """
    Directional spin mask — prevents spinning AWAY from the cylinder.

    When visible:
      - cylinder on RIGHT (cx > 0.55): block spin-left  (action 0), allow spin-right
      - cylinder on LEFT  (cx < 0.45): block spin-right (action 1), allow spin-left
      - cylinder centred  (|cx-0.5|<0.15): block BOTH spins (already centred, just approach)

    This lets the robot spin IN PLACE to re-centre a cylinder that appeared at
    the edge of frame, while still preventing overshooting past it.
    """
    visible = state_tensor[:, 3] > 0.5          # (batch,) bool
    if not visible.any():
        return logits

    cx           = state_tensor[:, 0]
    centre_error = (cx - 0.5).abs()
    masked       = logits.clone()

    # Cylinder on right → spinning left would lose it → block spin-left
    spin_left_bad  = visible & (cx > 0.53)
    masked[spin_left_bad,  0] = -1e9

    # Cylinder on left → spinning right would lose it → block spin-right
    spin_right_bad = visible & (cx < 0.47)
    masked[spin_right_bad, 1] = -1e9

    # Already centred → block both spins, just approach
    # In apply_spin_mask:
    centred = visible & (centre_error < 0.12)   # match the approach gate exactly
    masked[centred, 0] = -1e9
    masked[centred, 1] = -1e9

    return masked


class ActorCritic(nn.Module):
    """
    Shared MLP trunk → two heads:
      actor  — outputs logits over 5 actions
      critic — outputs a scalar state value
    """
    def __init__(self, state_dim=4, n_actions=5, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        f      = self.trunk(x)
        logits = self.actor(f)
        value  = self.critic(f).squeeze(-1)
        return logits, value

    def get_action(self, state_tensor):
        """Sample action during rollout collection (spins blocked when visible)."""
        logits, value = self.forward(state_tensor)
        logits   = apply_spin_mask(logits, state_tensor)   # ← mask applied here
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy()

    def evaluate(self, states, actions):
        """Re-evaluate for PPO update (same mask applied for consistency)."""
        logits, values = self.forward(states)
        logits    = apply_spin_mask(logits, states)         # ← same mask here
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs, values, dist.entropy()


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ROLLOUT BUFFER  (stores one on-policy rollout)
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, device):
        """Generalised Advantage Estimation → returns tensors for PPO update."""
        n          = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        vals       = np.array([v.item() for v in self.values] + [last_value],
                               dtype=np.float32)

        for t in reversed(range(n)):
            delta = self.rewards[t] + GAMMA * vals[t+1] * (1 - self.dones[t]) - vals[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + vals[:n]

        states    = torch.FloatTensor(np.array(self.states)).to(device)
        actions   = torch.LongTensor(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).to(device).detach()
        adv_t     = torch.FloatTensor(advantages).to(device)
        ret_t     = torch.FloatTensor(returns).to(device)

        # Normalise advantages for training stability
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        return states, actions, log_probs, adv_t, ret_t

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones     = [], [], []

    def __len__(self):
        return len(self.rewards)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PPO UPDATE
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(policy, optimizer, states, actions, old_log_probs,
               advantages, returns, device):
    n = len(states)
    pl_sum, vl_sum, ent_sum = 0.0, 0.0, 0.0

    for _ in range(N_EPOCHS):
        idx = torch.randperm(n)

        for start in range(0, n, MINI_BATCH):
            mb = idx[start: start + MINI_BATCH]

            new_lp, vals, entropy = policy.evaluate(states[mb], actions[mb])

            ratio = torch.exp(new_lp - old_log_probs[mb])
            adv   = advantages[mb]

            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            p_loss = -torch.min(surr1, surr2).mean()

            v_loss = nn.functional.mse_loss(vals, returns[mb])
            e_loss = -entropy.mean()

            loss = p_loss + VALUE_COEF * v_loss + ENTROPY_COEF * e_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()

            pl_sum  += p_loss.item()
            vl_sum  += v_loss.item()
            ent_sum += entropy.mean().item()

    n_updates = N_EPOCHS * max(1, n // MINI_BATCH)
    return pl_sum / n_updates, vl_sum / n_updates, ent_sum / n_updates


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env       = HuskyEnv(gui=False)
    policy    = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    buffer    = RolloutBuffer()

    ep_rewards, wins = [], []
    best_avg         = -float("inf")

    # Episode accumulators
    ep_reward, ep_steps, ep_won, ep_max_area = 0.0, 0, False, 0.0
    ep_count, total_steps = 0, 0
    last_pl, last_vl, last_ent = 0.0, 0.0, 0.0

    state = env.reset()

    print(f"\nStarting PPO3 — rollout={ROLLOUT_STEPS}, epochs={N_EPOCHS}, "
          f"batch={MINI_BATCH}")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'MaxArea':>7} {'BestAvg':>8}  Result")

    while ep_count < MAX_EPISODES:

        # ── Collect rollout ──────────────────────────────────────────────────
        for _ in range(ROLLOUT_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value, _ = policy.get_action(s_t)

            next_state, reward, done = env.step(action.item())

            buffer.store(
                state,
                action.item(),
                log_prob.squeeze(0),
                reward,
                value.squeeze(0),
                float(done),
            )

            state        = next_state
            ep_reward   += reward
            ep_steps    += 1
            total_steps += 1
            if state[3]:                        # visible
                ep_max_area = max(ep_max_area, float(state[2]))

            if done or ep_steps >= MAX_STEPS:
                ep_won   = done
                ep_count += 1
                ep_rewards.append(ep_reward)
                wins.append(int(ep_won))

                avg      = np.mean(ep_rewards[-20:])
                win_rate = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)

                if avg > best_avg:
                    best_avg = avg
                    torch.save(policy.state_dict(), "husky_ppo3_best.pth")

                tag = "WIN" if ep_won else "   "
                print(f"{ep_count:4d} {ep_reward:8.2f} {avg:8.2f} "
                      f"{win_rate*100:6.1f} {ep_steps:5d} {ep_max_area:7.3f} "
                      f"{best_avg:8.2f}  {tag}")

                if ep_count % 50 == 0:
                    ckpt = f"husky_ppo3_ep{ep_count}.pth"
                    torch.save(policy.state_dict(), ckpt)
                    print(f"  → checkpoint {ckpt}  "
                          f"[π {last_pl:.4f}  V {last_vl:.4f}  H {last_ent:.3f}"
                          f"  steps {total_steps}]")

                if len(wins) >= 50 and win_rate >= EARLY_STOP_RATE:
                    print(f"\nEarly stop: win rate {win_rate*100:.1f}% "
                          f"over last 50 episodes.")
                    env.close()
                    torch.save(policy.state_dict(), "husky_ppo3_final.pth")
                    print("Saved: husky_ppo3_final.pth  husky_ppo3_best.pth")
                    return policy

                if ep_count >= MAX_EPISODES:
                    break

                state = env.reset()
                ep_reward, ep_steps, ep_won, ep_max_area = 0.0, 0, False, 0.0

        # ── Bootstrap & PPO update ───────────────────────────────────────────
        with torch.no_grad():
            s_t         = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, last_val, _, _ = policy.get_action(s_t)
            last_value = 0.0 if buffer.dones[-1] == 1.0 else last_val.item()

        states, actions, old_lp, adv, ret = buffer.compute_gae(last_value, device)
        last_pl, last_vl, last_ent = ppo_update(
            policy, optimizer, states, actions, old_lp, adv, ret, device)
        buffer.clear()

        print(f"  ↻ update | π {last_pl:.4f}  V {last_vl:.4f}  "
              f"H {last_ent:.3f}  steps {total_steps}")

    env.close()
    torch.save(policy.state_dict(), "husky_ppo3_final.pth")
    print("Done. Saved: husky_ppo3_final.pth  husky_ppo3_best.pth")
    return policy


# ═══════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE  (watch the trained agent in GUI)
# ═══════════════════════════════════════════════════════════════════════════

def run_trained(model_path="husky_ppo3_best.pth", n_episodes=10):
    device = torch.device("cpu")
    policy = ActorCritic().to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    policy.eval()
    print(f"Loaded {model_path}")

    env          = HuskyEnv(gui=True)
    ACTION_NAMES = ["spin-L", "spin-R", "fwd  ", "fwd+L", "fwd+R"]
    wins         = 0

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0

        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(s_t)
                logits    = apply_spin_mask(logits, s_t)   # ← fix: mask at inference
                action    = logits.argmax(dim=-1).item()   # greedy

            state, reward, done = env.step(action)
            total_r += reward
            cx, _, area, vis = state

            print(f"ep {ep:2d} step {step:3d} | {ACTION_NAMES[action]}"
                  f" | cx={cx:.2f} area={area:.3f} vis={int(vis)}"
                  f" | r={reward:+.2f}")

            time.sleep(1 / 60)

            if done:
                wins += 1
                print(f"  *** WIN!  ep_reward={total_r:.1f} ***\n")
                time.sleep(2)
                break
        else:
            print(f"  ep {ep} timeout.  ep_reward={total_r:.1f}\n")

    env.close()
    print(f"\nWin rate: {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        path = sys.argv[2] if len(sys.argv) > 2 else "husky_ppo3_final.pth"
        run_trained(model_path=path)
    else:
        train()