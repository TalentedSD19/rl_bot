"""
Husky DQN — Camera-only box-chasing agent
==========================================
Task: Use only the camera feed to:
  1. Spin to find the yellow box if it's not visible
  2. Centre the box horizontally
  3. Drive toward it until it fills >50% of the screen

Actions (5 discrete):
  0 = spin left
  1 = spin right
  2 = forward
  3 = forward + steer left
  4 = forward + steer right

State (4 floats, all in [0, 1]):
  [cx, cy, area, visible]
  All zero when box not detected.

Reward per step:
  +100      win: box fills >50% of frame (episode ends)
  delta*20  approach reward — proportional to area increase (negative if retreating)
  -ce*2     continuous centre-error penalty when visible
  +1.0      forward action when box is centred (ce < 0.15)
  +0.5      steering toward an off-centre box
  -0.5      steering away from the box
  -0.3      spinning in place while box is visible
  -0.5      driving forward while box is not visible
  -0.1      time pressure per step

Usage:
  python husky_dqn2.py            # train
  python husky_dqn2.py run        # run best checkpoint
  python husky_dqn2.py run my.pth # run a specific checkpoint
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Simulation constants ────────────────────────────────────────────────────
SPEED        = 15
LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]

# Camera
CAM_W, CAM_H = 320, 240
FOV          = 90

# Box detection: yellow pixel = R>180, G>150, B<80
YELLOW_R_MIN, YELLOW_G_MIN, YELLOW_B_MAX = 180, 150, 80
MIN_YELLOW_PIXELS = 5     # ignore single-pixel noise

# Training
MAX_EPISODES        = 800
MAX_STEPS           = 200
BATCH_SIZE          = 64
WARMUP_STEPS        = 1000   # random transitions before any gradient update
GAMMA               = 0.99
LR                  = 1e-3
EPS_START           = 1.0
EPS_END             = 0.05
EPS_DECAY           = 0.995  # per episode; reaches 0.05 around ep 600
TARGET_UPDATE       = 5      # episodes between hard target-network copies
MEMORY_SIZE         = 20_000
GRAD_CLIP           = 10.0
WIN_AREA_FRAC       = 0.50
EARLY_STOP_WIN_RATE = 0.80   # stop if win rate > 80% over last 50 episodes


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class HuskyEnv:

    def __init__(self, gui=False):
        mode = p.GUI if gui else p.DIRECT
        self.client       = p.connect(mode)
        self.prev_visible = False
        self.prev_area    = 0.0
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        self._setup_sim()

    def _setup_sim(self):
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1],
                                physicsClientId=self.client)
        self._build_box()

    def _build_box(self):
        half = [0.5, 0.5, 0.5]
        col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                      physicsClientId=self.client)
        vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                   rgbaColor=[1.0, 0.86, 0.0, 1.0],
                                   specularColor=[0, 0, 0],
                                   physicsClientId=self.client)
        self.box_id = p.createMultiBody(
            baseMass=5,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[3, 0, 0.5],
            physicsClientId=self.client,
        )

    def _place_box_random(self):
        # Use np.random.uniform so global seed (np.random.seed) is respected
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(3.0, 6.0)
        pos   = [dist * np.cos(angle), dist * np.sin(angle), 0.5]
        p.resetBasePositionAndOrientation(
            self.box_id, pos, [0, 0, 0, 1],
            physicsClientId=self.client,
        )
        # Zero out any residual velocity from the previous episode
        p.resetBaseVelocity(
            self.box_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client,
        )

    def _drive(self, left_vel, right_vel):
        for j in LEFT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=left_vel, force=100,
                                    physicsClientId=self.client)
        for j in RIGHT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=right_vel, force=100,
                                    physicsClientId=self.client)

    def _get_rgb(self):
        pos, ori = p.getBasePositionAndOrientation(self.husky,
                                                   physicsClientId=self.client)
        rot    = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        offset = np.array([0.5, 0.0, 0.3])
        eye    = np.array(pos) + rot @ offset
        target = eye + rot @ np.array([1.0, 0.0, 0.0])
        up     = rot @ np.array([0.0, 0.0, 1.0])
        # computeViewMatrix / computeProjectionMatrixFOV are pure math; no clientId
        view   = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj   = p.computeProjectionMatrixFOV(FOV, CAM_W / CAM_H, 0.1, 20.0)
        _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=self.client)
        return np.array(rgb, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]

    @staticmethod
    def detect_box(rgb):
        mask = (
            (rgb[:, :, 0] > YELLOW_R_MIN) &
            (rgb[:, :, 1] > YELLOW_G_MIN) &
            (rgb[:, :, 2] < YELLOW_B_MAX)
        )
        n = int(mask.sum())
        if n < MIN_YELLOW_PIXELS:
            return None
        ys, xs = np.where(mask)
        cx   = xs.mean() / CAM_W
        cy   = ys.mean() / CAM_H
        area = n / (CAM_W * CAM_H)
        return (cx, cy, area)

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.husky, [0, 0, 0.1], [0, 0, 0, 1],
            physicsClientId=self.client,
        )
        # Zero out robot chassis velocity so it doesn't carry momentum
        p.resetBaseVelocity(self.husky, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.client)
        n_joints = p.getNumJoints(self.husky, physicsClientId=self.client)
        for j in range(n_joints):
            p.resetJointState(self.husky, j, 0, 0,
                              physicsClientId=self.client)
        self._drive(0, 0)
        self._place_box_random()
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)
        self.prev_visible = False
        self.prev_area    = 0.0
        return self._get_state()

    def _get_state(self):
        info = self.detect_box(self._get_rgb())
        if info is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cx, cy, area = info
        return np.array([cx, cy, area, 1.0], dtype=np.float32)

    def step(self, action):
        vel_map = {
            0: (-SPEED,         +SPEED),        # spin left
            1: (+SPEED,         -SPEED),        # spin right
            2: (+SPEED,         +SPEED),        # forward
            3: (+SPEED * 0.4,   +SPEED),        # forward + steer left
            4: (+SPEED,         +SPEED * 0.4),  # forward + steer right
        }
        self._drive(*vel_map[action])
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        state          = self._get_state()
        reward, done   = self._compute_reward(state, action)
        self.prev_visible = bool(state[3])
        if state[3]:                        # update only when box is visible
            self.prev_area = state[2]
        return state, reward, done

    def _compute_reward(self, state, action):
        cx, cy, area, visible = state
        visible = bool(visible)
        reward  = 0.0

        # ── box not visible ──────────────────────────────────────────────────
        if not visible:
            if action == 2:                 # blind forward — penalise
                reward -= 0.5
            elif action in (3, 4):          # blind steer-forward — mild penalty
                reward -= 0.2
            # spinning while blind: neutral (no reward, no penalty)

        # ── box visible ──────────────────────────────────────────────────────
        if visible:
            if area > WIN_AREA_FRAC:
                return reward + 100.0, True

            centre_error = abs(cx - 0.5)

            # Continuous gradient toward centre
            reward -= centre_error * 2.0

            if centre_error < 0.15:
                if action == 2:             # driving straight toward centred box
                    reward += 1.0
                elif action in (3, 4):      # unnecessary steer while centred
                    reward -= 0.3
            else:
                # Steer toward box
                if (cx < 0.5 and action == 3) or (cx > 0.5 and action == 4):
                    reward += 0.5
                # Steer away from box
                if (cx < 0.5 and action == 4) or (cx > 0.5 and action == 3):
                    reward -= 0.5

            # Spinning in place while box is visible
            if action in (0, 1):
                reward -= 0.3

            # Approach reward — positive when closing in, negative when retreating
            reward += (area - self.prev_area) * 20.0

        reward -= 0.1   # time pressure
        return reward, False

    def close(self):
        p.disconnect(self.client)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DQN NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class DQN(nn.Module):
    def __init__(self, state_dim=4, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buf.append((s, a, r, s_, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    env    = HuskyEnv(gui=False)
    policy = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory    = ReplayBuffer(MEMORY_SIZE)
    loss_fn   = nn.SmoothL1Loss()   # Huber loss — robust to large TD errors
    eps       = EPS_START

    episode_rewards = []
    wins            = []
    total_steps     = 0
    best_avg        = -float("inf")

    # ── Warmup: fill buffer with random transitions before training ──────────
    print(f"Warming up replay buffer ({WARMUP_STEPS} steps)...")
    state = env.reset()
    for _ in range(WARMUP_STEPS):
        action                   = random.randrange(5)
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, float(done))
        state = env.reset() if done else next_state
    print(f"  buffer size: {len(memory)}")

    # ── Main training loop ───────────────────────────────────────────────────
    for ep in range(1, MAX_EPISODES + 1):
        state   = env.reset()
        total_r = 0.0
        won     = False

        for _ in range(MAX_STEPS):
            if random.random() < eps:
                action = random.randrange(5)
            else:
                with torch.no_grad():
                    s_t    = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy(s_t).argmax().item()

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state       = next_state
            total_r    += reward
            total_steps += 1

            # ── Gradient update ──────────────────────────────────────────────
            s, a, r, s_, d = [x.to(device) for x in memory.sample(BATCH_SIZE)]

            q_pred = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double DQN: policy selects action, target evaluates it
                best_a = policy(s_).argmax(1, keepdim=True)
                q_next = target(s_).gather(1, best_a).squeeze(1)
                q_tgt  = r + GAMMA * q_next * (1 - d)

            loss = loss_fn(q_pred, q_tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()

            if done:
                won = True
                break

        eps = max(EPS_END, eps * EPS_DECAY)
        episode_rewards.append(total_r)
        wins.append(int(won))

        if ep % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        avg     = np.mean(episode_rewards[-20:])
        win_rate = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)

        print(f"Ep {ep:4d} | reward {total_r:8.2f} | avg(20) {avg:8.2f}"
              f" | win%50 {win_rate*100:5.1f} | eps {eps:.3f}"
              f" | {'WIN' if won else '   '}")

        if ep % 50 == 0:
            torch.save(policy.state_dict(), f"husky_dqn_ep{ep}.pth")
            print(f"  → checkpoint: husky_dqn_ep{ep}.pth")

        # Save best model whenever rolling average improves
        if avg > best_avg:
            best_avg = avg
            torch.save(policy.state_dict(), "husky_dqn_best.pth")

        # Early stopping
        if len(wins) >= 50 and win_rate >= EARLY_STOP_WIN_RATE:
            print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
            break

    env.close()
    torch.save(policy.state_dict(), "husky_dqn_final.pth")
    print("Training complete. Models saved: husky_dqn_final.pth, husky_dqn_best.pth")
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_trained(model_path="husky_dqn_best.pth", n_episodes=10):
    device = torch.device("cpu")
    policy = DQN().to(device)
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
            with torch.no_grad():
                s_t    = torch.FloatTensor(state).unsqueeze(0)
                action = policy(s_t).argmax().item()

            state, reward, done = env.step(action)
            total_r += reward
            cx, _, area, vis = state

            print(f"ep {ep:2d} step {step:3d} | {ACTION_NAMES[action]}"
                  f" | cx={cx:.2f} area={area:.3f} vis={int(vis)}"
                  f" | r={reward:+.2f}")

            time.sleep(1 / 60)

            if done:
                wins += 1
                print(f"  *** WIN! ep_reward={total_r:.1f} ***\n")
                time.sleep(2)
                break
        else:
            print(f"  ep {ep} timeout. ep_reward={total_r:.1f}\n")

    env.close()
    print(f"Win rate: {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        path = sys.argv[2] if len(sys.argv) > 2 else "husky_dqn_best.pth"
        run_trained(model_path=path)
    else:
        train()
