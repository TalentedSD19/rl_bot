"""
husky_task2_ppo.py  —  camera-only pick-and-place, trained with PPO
====================================================================

Task sequence
─────────────
  0  FIND_GREEN     spin until the small green cylinder is visible
  1  APPROACH_GREEN drive toward it, keep it centred, until close enough
  2  PICKUP         [auto] lower fork → magnet on → attach → lift to max
  3  FIND_RED       spin until the large red cylinder is visible
  4  APPROACH_RED   drive toward it, keep it centred, until close enough
  5  DROP           [auto] magnet off → reverse → lower fork  →  WIN

Observation  (10 floats)
  green_cx  green_cy  green_area  green_vis
  red_cx    red_cy    red_area    red_vis
  lift_norm  phase_norm

Actions  (5 discrete)  —  movement only; lift/magnet automated
  0 spin-L   1 spin-R   2 fwd   3 fwd+steer-L   4 fwd+steer-R

Usage
  python husky_task2_ppo.py           # train
  python husky_task2_ppo.py run       # watch best checkpoint
  python husky_task2_ppo.py run path  # watch specific checkpoint
"""

import sys, os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pybullet as p
import pybullet_data

# ── Paths ──────────────────────────────────────────────────────────────────────
SIM_DIR       = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulations'))
FORKLIFT_URDF = os.path.join(SIM_DIR, 'forklift_mast.urdf')

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Simulation constants ───────────────────────────────────────────────────────
SPEED        = 15
LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]
CAM_W, CAM_H = 320, 240
FOV          = 90
LIFT_JOINT   = 0
MAGNET_LINK  = 2
LIFT_STEP    = 0.02                    # lift_target delta per env step
LIFT_MIN, LIFT_MAX = -0.4, 0.4

# ── Object geometry (must match husky_task2.py) ────────────────────────────────
BIG_R,  BIG_H = 0.4,  0.5
SML_R,  SML_H = BIG_R / 2.5, BIG_H / 2.5   # 0.16, 0.20

# ── Phase constants ────────────────────────────────────────────────────────────
PHASE_FIND_GREEN     = 0
PHASE_APPROACH_GREEN = 1
PHASE_PICKUP         = 2   # automated
PHASE_FIND_RED       = 3
PHASE_APPROACH_RED   = 4
PHASE_DROP           = 5   # automated
N_PHASES             = 6

# ── Task thresholds ────────────────────────────────────────────────────────────
GREEN_CLOSE_AREA   = 0.030  # green area fraction that triggers PICKUP
RED_CLOSE_AREA     = 0.090  # red   area fraction that triggers DROP
MAGNET_DIST        = 0.5    # metres — match husky_task2.py
AUTO_REVERSE_STEPS = 40     # env steps to reverse after dropping

# ── PPO hyper-parameters ───────────────────────────────────────────────────────
MAX_EPISODES    = 2000
MAX_STEPS       = 700
ROLLOUT_STEPS   = 2048
N_EPOCHS        = 10
MINI_BATCH      = 64
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.05
VALUE_COEF      = 0.5
LR              = 3e-4
GRAD_CLIP       = 0.5
EARLY_STOP_RATE = 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class HuskyTask2Env:
    """Multi-phase pick-and-place. Front camera is the only sensor."""

    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        self.constraint_id = None
        self._build_world()

    # ── World setup ───────────────────────────────────────────────────────────

    def _build_world(self):
        pid = self.client
        p.setGravity(0, 0, -9.81, physicsClientId=pid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=pid)
        self.husky    = p.loadURDF("husky/husky.urdf", [0, 0, 0.15],
                                   physicsClientId=pid)
        self.mast     = p.loadURDF(FORKLIFT_URDF, basePosition=[0.6, 0, 0.8],
                                   useFixedBase=False, physicsClientId=pid)

        weld = p.createConstraint(self.husky, -1, self.mast, -1,
                                  p.JOINT_FIXED, [0, 0, 0],
                                  [0.6, 0, -0.15], [0, 0, -0.8],
                                  physicsClientId=pid)
        p.changeConstraint(weld, maxForce=50000, physicsClientId=pid)

        def make_cyl(mass, r, h, color):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=pid)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=color, physicsClientId=pid)
            return p.createMultiBody(mass, col, vis, physicsClientId=pid)

        self.big_cyl   = make_cyl(0,   BIG_R, BIG_H, [1.0, 0.0, 0.0, 1.0])
        self.small_cyl = make_cyl(0.2, SML_R, SML_H, [0.0, 1.0, 0.0, 1.0])

    def _rand_pos(self, lo, hi, z):
        a, d = np.random.uniform(0, 2*np.pi), np.random.uniform(lo, hi)
        return [d*np.cos(a), d*np.sin(a), z]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        pid = self.client

        # Reset husky
        p.resetBasePositionAndOrientation(self.husky, [0,0,0.15], [0,0,0,1],
                                          physicsClientId=pid)
        p.resetBaseVelocity(self.husky, [0,0,0], [0,0,0], physicsClientId=pid)
        for j in range(p.getNumJoints(self.husky, physicsClientId=pid)):
            p.resetJointState(self.husky, j, 0, 0, physicsClientId=pid)

        # Reset mast
        p.resetBasePositionAndOrientation(self.mast, [0.6,0,0.8], [0,0,0,1],
                                          physicsClientId=pid)
        p.resetBaseVelocity(self.mast, [0,0,0], [0,0,0], physicsClientId=pid)
        p.resetJointState(self.mast, LIFT_JOINT, 0, 0, physicsClientId=pid)

        # Clean up magnet
        if self.constraint_id is not None:
            try:
                p.removeConstraint(self.constraint_id, physicsClientId=pid)
            except Exception:
                pass
            self.constraint_id = None
        p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                 enableCollision=1, physicsClientId=pid)
        p.changeVisualShape(self.mast, MAGNET_LINK,
                            rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=pid)

        # Random object placement
        p.resetBasePositionAndOrientation(
            self.big_cyl, self._rand_pos(3.0, 6.0, BIG_H/2), [0,0,0,1],
            physicsClientId=pid)
        p.resetBasePositionAndOrientation(
            self.small_cyl, self._rand_pos(2.0, 4.0, SML_H/2), [0,0,0,1],
            physicsClientId=pid)

        # Internal state
        self.phase         = PHASE_FIND_GREEN
        self.lift_target   = 0.0
        self.step_count    = 0
        self.auto_sub      = 0
        self.auto_counter  = 0
        self.magnet_on     = False

        self._reset_nav_trackers()

        for _ in range(20):
            p.stepSimulation(physicsClientId=pid)

        return self._get_obs()

    # ── Actuation ─────────────────────────────────────────────────────────────

    def _drive(self, left, right):
        pid = self.client
        for j in LEFT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=left, force=100,
                                    physicsClientId=pid)
        for j in RIGHT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=right, force=100,
                                    physicsClientId=pid)

    def _update_lift(self):
        p.setJointMotorControl2(self.mast, LIFT_JOINT, p.POSITION_CONTROL,
                                targetPosition=self.lift_target, force=2000,
                                physicsClientId=self.client)

    # ── Magnet ────────────────────────────────────────────────────────────────

    def _magnet_pos(self):
        return np.array(p.getLinkState(self.mast, MAGNET_LINK,
                                       physicsClientId=self.client)[0])

    def _try_attach(self):
        if self.constraint_id is not None:
            return True
        cyl_pos = np.array(p.getBasePositionAndOrientation(
            self.small_cyl, physicsClientId=self.client)[0])
        if np.linalg.norm(self._magnet_pos() - cyl_pos) < MAGNET_DIST:
            self.constraint_id = p.createConstraint(
                self.mast, MAGNET_LINK, self.small_cyl, -1,
                p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0],
                physicsClientId=self.client)
            p.changeConstraint(self.constraint_id, maxForce=500000,
                               physicsClientId=self.client)
            p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                     enableCollision=0, physicsClientId=self.client)
            return True
        return False

    def _detach(self):
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id, physicsClientId=self.client)
            self.constraint_id = None
            p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                     enableCollision=1, physicsClientId=self.client)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _get_rgb(self):
        pid    = self.client
        pos, ori = p.getBasePositionAndOrientation(self.husky, physicsClientId=pid)
        rot    = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        eye    = np.array(pos) + rot @ [0.5, 0.0, 0.3]
        target = eye + rot @ [1.0, 0.0, 0.0]
        up     = rot @ [0.0, 0.0, 1.0]
        view   = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj   = p.computeProjectionMatrixFOV(FOV, CAM_W/CAM_H, 0.02, 20.0)
        _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=pid)
        return np.array(rgb, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]

    @staticmethod
    def _detect(rgb, green=True):
        """Returns (cx, cy, area) normalised to frame, or None."""
        if green:
            mask = (rgb[:,:,0] < 80) & (rgb[:,:,1] > 150) & (rgb[:,:,2] < 80)
        else:
            mask = (rgb[:,:,0] > 150) & (rgb[:,:,1] < 80)  & (rgb[:,:,2] < 80)
        n = int(mask.sum())
        if n < 5:
            return None
        ys, xs = np.where(mask)
        return xs.mean()/CAM_W, ys.mean()/CAM_H, n/(CAM_W*CAM_H)

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        rgb = self._get_rgb()
        g   = self._detect(rgb, green=True)
        r   = self._detect(rgb, green=False)
        gobs = list(g) + [1.0] if g else [0.0, 0.0, 0.0, 0.0]
        robs = list(r) + [1.0] if r else [0.0, 0.0, 0.0, 0.0]
        lift = p.getJointState(self.mast, LIFT_JOINT,
                               physicsClientId=self.client)[0]
        lift_norm  = (lift - LIFT_MIN) / (LIFT_MAX - LIFT_MIN)
        phase_norm = self.phase / (N_PHASES - 1)
        return np.array(gobs + robs + [lift_norm, phase_norm], dtype=np.float32)

    def _active_obs(self, obs):
        """(cx, cy, area, vis) for the target of the current nav phase."""
        if self.phase in (PHASE_FIND_GREEN, PHASE_APPROACH_GREEN):
            return obs[0], obs[1], obs[2], obs[3]
        return obs[4], obs[5], obs[6], obs[7]

    # ── Nav trackers ──────────────────────────────────────────────────────────

    def _reset_nav_trackers(self):
        self.prev_visible       = False
        self.prev_area          = 0.0
        self.prev_centre_error  = 0.5
        self.discovery_rewarded = False

    def _update_nav_trackers(self, obs):
        cx, _, area, visible = self._active_obs(obs)
        self.prev_visible = bool(visible)
        if visible:
            self.prev_area         = area
            self.prev_centre_error = abs(cx - 0.5)

    # ── Step ──────────────────────────────────────────────────────────────────

    VEL_MAP = {
        0: (-SPEED,       +SPEED),
        1: (+SPEED,       -SPEED),
        2: (+SPEED,       +SPEED),
        3: (+SPEED * 0.4, +SPEED),
        4: (+SPEED,       +SPEED * 0.4),
    }

    def step(self, action):
        self.step_count += 1

        if self.phase in (PHASE_PICKUP, PHASE_DROP):
            return self._step_automated()

        self._drive(*self.VEL_MAP[action])
        for _ in range(4):
            self._update_lift()
            p.stepSimulation(physicsClientId=self.client)

        obs            = self._get_obs()
        reward, done   = self._reward(obs, action)
        self._update_nav_trackers(obs)
        return obs, reward, done

    # ── Automated phases ──────────────────────────────────────────────────────

    def _step_automated(self):
        pid    = self.client
        reward = 0.0
        done   = False

        self._drive(0, 0)

        # ── PICKUP ────────────────────────────────────────────────────────────
        if self.phase == PHASE_PICKUP:

            if self.auto_sub == 0:                          # lower fork
                self.lift_target = max(self.lift_target - LIFT_STEP, LIFT_MIN)
                if self.lift_target <= LIFT_MIN + 0.01:
                    self.auto_sub     = 1
                    self.auto_counter = 0

            elif self.auto_sub == 1:                        # enable magnet & attach
                if not self.magnet_on:
                    self.magnet_on = True
                    p.changeVisualShape(self.mast, MAGNET_LINK,
                                        rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                        physicsClientId=pid)
                if self._try_attach():
                    reward         += 5.0
                    self.auto_sub   = 2
                else:
                    self.auto_counter += 1
                    if self.auto_counter > 15:              # failed → retry approach
                        reward -= 3.0
                        self.magnet_on = False
                        p.changeVisualShape(self.mast, MAGNET_LINK,
                                            rgbaColor=[0.7, 0.7, 0.7, 1.0],
                                            physicsClientId=pid)
                        self.phase    = PHASE_APPROACH_GREEN
                        self.auto_sub = 0
                        self._reset_nav_trackers()

            elif self.auto_sub == 2:                        # lift to max
                self.lift_target = min(self.lift_target + LIFT_STEP, LIFT_MAX)
                if self.lift_target >= LIFT_MAX - 0.01:
                    reward += 10.0
                    self.phase    = PHASE_FIND_RED
                    self.auto_sub = 0
                    self._reset_nav_trackers()

        # ── DROP ──────────────────────────────────────────────────────────────
        elif self.phase == PHASE_DROP:

            if self.auto_sub == 0:                          # release
                self._detach()
                self.magnet_on = False
                p.changeVisualShape(self.mast, MAGNET_LINK,
                                    rgbaColor=[0.7, 0.7, 0.7, 1.0],
                                    physicsClientId=pid)
                reward         += 5.0
                self.auto_sub   = 1
                self.auto_counter = 0

            elif self.auto_sub == 1:                        # reverse
                self._drive(-SPEED, -SPEED)
                self.auto_counter += 1
                if self.auto_counter >= AUTO_REVERSE_STEPS:
                    self.auto_sub = 2

            elif self.auto_sub == 2:                        # lower fork to neutral
                self.lift_target = max(self.lift_target - LIFT_STEP, 0.0)
                if self.lift_target <= 0.01:
                    reward += 100.0
                    done    = True

        for _ in range(4):
            self._update_lift()
            p.stepSimulation(physicsClientId=pid)

        return self._get_obs(), reward, done

    # ── Reward ────────────────────────────────────────────────────────────────

    def _reward(self, obs, action):
        cx, cy, area, visible = self._active_obs(obs)
        visible = bool(visible)
        r       = 0.0

        if not visible:
            if self.prev_visible:
                r -= 5.0                   # lost the target
            if action == 2:
                r -= 1.0                   # blind forward
            elif action in (3, 4):
                r -= 0.5
            r -= 0.05
            return r, False

        # One-time discovery bonus per phase
        if not self.prev_visible and not self.discovery_rewarded:
            r += 3.0
            self.discovery_rewarded = True

        centre_error = abs(cx - 0.5)

        # Centering reward
        r += (self.prev_centre_error - centre_error) * 5.0
        r -= centre_error * 0.5

        # Directional steering bonus
        if cx > 0.53 and action in (1, 4):
            r += 0.4
        elif cx < 0.47 and action in (0, 3):
            r += 0.4

        # Phase-specific: FIND → transition once visible
        if self.phase in (PHASE_FIND_GREEN, PHASE_FIND_RED):
            if centre_error < 0.30:
                self.phase = (PHASE_APPROACH_GREEN if self.phase == PHASE_FIND_GREEN
                              else PHASE_APPROACH_RED)
                r += 1.0

        # Phase-specific: APPROACH → reward area growth, trigger automation
        elif self.phase in (PHASE_APPROACH_GREEN, PHASE_APPROACH_RED):
            close_area = (GREEN_CLOSE_AREA if self.phase == PHASE_APPROACH_GREEN
                          else RED_CLOSE_AREA)

            if centre_error < 0.12:
                delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
                r += delta_area * 30.0
            elif centre_error < 0.20:
                delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
                r += delta_area * 10.0
            else:
                if action in (2, 3, 4):
                    r -= 0.4

            if area >= close_area and centre_error < 0.20:
                r += 10.0
                self.phase    = (PHASE_PICKUP if self.phase == PHASE_APPROACH_GREEN
                                 else PHASE_DROP)
                self.auto_sub     = 0
                self.auto_counter = 0

        r -= 0.05
        return r, False

    def close(self):
        p.disconnect(self.client)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ACTOR-CRITIC
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, state_dim=10, n_actions=5, hidden=128):
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
        logits, value = self(s)
        logits  = apply_spin_mask(logits, s)
        dist    = Categorical(logits=logits)
        action  = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, states, actions):
        logits, values = self(states)
        logits    = apply_spin_mask(logits, states)
        dist      = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


def apply_spin_mask(logits, states):
    """
    Block spinning away from the active target.
    Uses green obs (cols 0,3) for phases 0-1, red obs (cols 4,7) for phases 3-4.
    """
    phase = (states[:, 9] * (N_PHASES - 1)).round().long()
    masked = logits.clone()

    for is_green in (True, False):
        target_phase_lo = PHASE_FIND_GREEN    if is_green else PHASE_FIND_RED
        target_phase_hi = PHASE_APPROACH_GREEN if is_green else PHASE_APPROACH_RED
        cx_col, vis_col = (0, 3) if is_green else (4, 7)

        active = ((phase == target_phase_lo) | (phase == target_phase_hi))
        vis    = active & (states[:, vis_col] > 0.5)
        if not vis.any():
            continue
        cx           = states[:, cx_col]
        centre_error = (cx - 0.5).abs()

        masked[vis & (cx > 0.53), 0] = -1e9   # cylinder on right → block spin-L
        masked[vis & (cx < 0.47), 1] = -1e9   # cylinder on left  → block spin-R
        masked[vis & (centre_error < 0.12), 0] = -1e9   # centred → no spinning
        masked[vis & (centre_error < 0.12), 1] = -1e9

    return masked


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ROLLOUT BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state);    self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value);    self.dones.append(done)

    def compute_gae(self, last_value, device):
        n          = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        vals       = np.array([v.item() for v in self.values] + [last_value],
                               dtype=np.float32)
        for t in reversed(range(n)):
            delta         = self.rewards[t] + GAMMA*vals[t+1]*(1-self.dones[t]) - vals[t]
            gae           = delta + GAMMA*GAE_LAMBDA*(1-self.dones[t])*gae
            advantages[t] = gae
        returns   = advantages + vals[:n]
        states    = torch.FloatTensor(np.array(self.states)).to(device)
        actions   = torch.LongTensor(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).to(device).detach()
        adv_t     = torch.FloatTensor(advantages).to(device)
        ret_t     = torch.FloatTensor(returns).to(device)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return states, actions, log_probs, adv_t, ret_t

    def clear(self):
        self.states = []; self.actions = []; self.log_probs = []
        self.rewards = []; self.values = []; self.dones = []

    def __len__(self):
        return len(self.rewards)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PPO UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

def ppo_update(policy, optimizer, states, actions, old_log_probs,
               advantages, returns, device):
    n = len(states)
    pl_sum = vl_sum = ent_sum = 0.0

    for _ in range(N_EPOCHS):
        idx = torch.randperm(n)
        for start in range(0, n, MINI_BATCH):
            mb = idx[start: start + MINI_BATCH]
            new_lp, vals, entropy = policy.evaluate(states[mb], actions[mb])
            ratio  = torch.exp(new_lp - old_log_probs[mb])
            adv    = advantages[mb]
            p_loss = -torch.min(ratio*adv,
                                torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)*adv).mean()
            v_loss = nn.functional.mse_loss(vals, returns[mb])
            loss   = p_loss + VALUE_COEF*v_loss - ENTROPY_COEF*entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()
            pl_sum  += p_loss.item()
            vl_sum  += v_loss.item()
            ent_sum += entropy.mean().item()

    n_up = N_EPOCHS * max(1, n // MINI_BATCH)
    return pl_sum/n_up, vl_sum/n_up, ent_sum/n_up


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env       = HuskyTask2Env(gui=False)
    policy    = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    buffer    = RolloutBuffer()

    ep_rewards, wins   = [], []
    best_avg           = -float("inf")
    ep_reward, ep_steps, ep_won = 0.0, 0, False
    ep_count, total_steps = 0, 0
    last_pl = last_vl = last_ent = 0.0

    state = env.reset()

    PHASE_NAMES = ["find-G", "appr-G", "pickup", "find-R", "appr-R", "drop  "]

    print(f"\nStarting  rollout={ROLLOUT_STEPS}  epochs={N_EPOCHS}  batch={MINI_BATCH}")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'Phase':>7} {'BestAvg':>8}  Result")

    while ep_count < MAX_EPISODES:

        for _ in range(ROLLOUT_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value, _ = policy.get_action(s_t)

            next_state, reward, done = env.step(action.item())
            buffer.store(state, action.item(), log_prob.squeeze(0),
                         reward, value.squeeze(0), float(done))

            state        = next_state
            ep_reward   += reward
            ep_steps    += 1
            total_steps += 1

            if done or ep_steps >= MAX_STEPS:
                ep_won    = done
                ep_count += 1
                ep_rewards.append(ep_reward)
                wins.append(int(ep_won))

                avg      = np.mean(ep_rewards[-20:])
                win_rate = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
                phase_tag = PHASE_NAMES[env.phase]

                if avg > best_avg:
                    best_avg = avg
                    torch.save(policy.state_dict(), "husky_task2_ppo_best.pth")

                tag = "WIN" if ep_won else "   "
                print(f"{ep_count:4d} {ep_reward:8.2f} {avg:8.2f} "
                      f"{win_rate*100:6.1f} {ep_steps:5d} {phase_tag:>7} "
                      f"{best_avg:8.2f}  {tag}")

                if ep_count % 100 == 0:
                    ckpt = f"husky_task2_ppo_ep{ep_count}.pth"
                    torch.save(policy.state_dict(), ckpt)
                    print(f"  → checkpoint {ckpt}  "
                          f"[π {last_pl:.4f}  V {last_vl:.4f}  "
                          f"H {last_ent:.3f}  steps {total_steps}]")

                if len(wins) >= 50 and win_rate >= EARLY_STOP_RATE:
                    print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
                    env.close()
                    torch.save(policy.state_dict(), "husky_task2_ppo_final.pth")
                    print("Saved: husky_task2_ppo_final.pth  husky_task2_ppo_best.pth")
                    return policy

                if ep_count >= MAX_EPISODES:
                    break

                state = env.reset()
                ep_reward, ep_steps, ep_won = 0.0, 0, False

        # Bootstrap + PPO update
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, last_val, _, _ = policy.get_action(s_t)
            last_value = 0.0 if buffer.dones[-1] == 1.0 else last_val.item()

        states, actions, old_lp, adv, ret = buffer.compute_gae(last_value, device)
        last_pl, last_vl, last_ent = ppo_update(
            policy, optimizer, states, actions, old_lp, adv, ret, device)
        buffer.clear()
        print(f"  ↻ update | π {last_pl:.4f}  V {last_vl:.4f}  "
              f"H {last_ent:.3f}  steps {total_steps}")

    env.close()
    torch.save(policy.state_dict(), "husky_task2_ppo_final.pth")
    print("Done. Saved: husky_task2_ppo_final.pth  husky_task2_ppo_best.pth")
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_trained(model_path="husky_task2_ppo_best.pth", n_episodes=5):
    device = torch.device("cpu")
    policy = ActorCritic().to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    policy.eval()
    print(f"Loaded {model_path}")

    env          = HuskyTask2Env(gui=True)
    ACTION_NAMES = ["spin-L", "spin-R", "fwd   ", "fwd+L ", "fwd+R "]
    PHASE_NAMES  = ["find-G", "appr-G", "pickup", "find-R", "appr-R", "drop  "]
    wins         = 0

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0

        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(s_t)
                logits    = apply_spin_mask(logits, s_t)
                action    = logits.argmax(dim=-1).item()

            state, reward, done = env.step(action)
            total_r += reward

            phase_name = PHASE_NAMES[env.phase]
            g_vis, r_vis = bool(state[3]), bool(state[7])
            print(f"ep {ep:2d} step {step:3d} | {phase_name} "
                  f"| {ACTION_NAMES[action]} "
                  f"| g_vis={int(g_vis)} g_area={state[2]:.3f} "
                  f"| r_vis={int(r_vis)} r_area={state[6]:.3f} "
                  f"| r={reward:+.2f}")

            time.sleep(1/60)

            if done:
                wins += 1
                print(f"  *** WIN!  ep_reward={total_r:.1f} ***\n")
                time.sleep(2)
                break
        else:
            print(f"  ep {ep} timeout  ep_reward={total_r:.1f}\n")

    env.close()
    print(f"\nWin rate: {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        path = sys.argv[2] if len(sys.argv) > 2 else "husky_task2_ppo_final.pth"
        run_trained(model_path=path)
    else:
        train()
