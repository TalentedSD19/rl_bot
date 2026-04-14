"""
Husky PPO — Camera-only box-chasing agent
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

PPO specifics:
  - On-policy rollout collection (ROLLOUT_STEPS per update)
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Shared Actor-Critic network with separate heads
  - Entropy bonus for exploration
  - Multiple mini-batch epochs per rollout

Usage:
  python husky_ppo2.py            # train
  python husky_ppo2.py run        # run best checkpoint
  python husky_ppo2.py run my.pth # run a specific checkpoint
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
MIN_YELLOW_PIXELS = 5

# ─── PPO Hyperparameters ─────────────────────────────────────────────────────
MAX_EPISODES        = 800
MAX_STEPS           = 200
ROLLOUT_STEPS       = 2048      # steps to collect before each PPO update
N_EPOCHS            = 10        # gradient epochs per rollout
MINI_BATCH_SIZE     = 64        # mini-batch size within each epoch
GAMMA               = 0.99      # discount factor
GAE_LAMBDA          = 0.95      # GAE smoothing parameter
CLIP_EPS            = 0.2       # PPO clipping epsilon
ENTROPY_COEF        = 0.05      # entropy bonus coefficient
VALUE_COEF          = 0.5       # value loss coefficient
LR                  = 3e-4      # Adam learning rate
GRAD_CLIP           = 0.5       # max gradient norm
WIN_AREA_FRAC       = 0.50
EARLY_STOP_WIN_RATE = 0.80      # stop if win rate > 80% over last 50 episodes


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
        # Force full opacity — tiny renderer can silently reset alpha on reload
        p.changeVisualShape(self.box_id, -1,
                            rgbaColor=[1.0, 0.86, 0.0, 1.0],
                            physicsClientId=self.client)

    def _place_box_random(self):
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(3.0, 6.0)
        pos   = [dist * np.cos(angle), dist * np.sin(angle), 0.5]
        p.resetBasePositionAndOrientation(
            self.box_id, pos, [0, 0, 0, 1],
            physicsClientId=self.client,
        )
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
        view   = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj   = p.computeProjectionMatrixFOV(FOV, CAM_W / CAM_H, 0.02, 20.0)
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
        self.step_count   = 0
        return self._get_state()

    def _get_state(self):
        info = self.detect_box(self._get_rgb())
        # Normalised step count in [0, 1] so the network knows how long it's searched
        t = self.step_count / MAX_STEPS
        if info is None:
            return np.array([0.0, 0.0, 0.0, 0.0, t], dtype=np.float32)
        cx, cy, area = info
        return np.array([cx, cy, area, 1.0, t], dtype=np.float32)

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
        self.step_count += 1
        state          = self._get_state()
        reward, done   = self._compute_reward(state, action)
        self.prev_visible = bool(state[3])
        if state[3]:
            self.prev_area = state[2]
        return state, reward, done

    def _compute_reward(self, state, action):
        cx, cy, area, visible, _ = state
        visible = bool(visible)
        reward  = 0.0

        # ── Box not visible ──────────────────────────────────────────────────
        if not visible:
            if action == 2:             # driving blind: likely to miss the box
                reward -= 0.5
            elif action in (3, 4):      # steer-forward blind: mild penalty
                reward -= 0.2
            # spinning blind: neutral — time pressure alone creates urgency

        # ── Discovery bonus: one-time reward for finding the box ─────────────
        if visible and not self.prev_visible:
            reward += 2.0

        # ── Box visible ──────────────────────────────────────────────────────
        if visible:
            if area > WIN_AREA_FRAC:
                return reward + 100.0, True

            centre_error = abs(cx - 0.5)

            # Smooth centering gradient: 0 at perfect center, -1.0 at screen edge.
            # Single continuous signal — no discrete action bonuses that conflict.
            reward -= centre_error * 2.0

            # Approach reward: only fires when box is reasonably centred so the
            # robot doesn't learn to charge at an off-centre box and overshoot.
            # delta_area is clipped to avoid a large spike on the first visible
            # step (when prev_area is still 0 from the last reset).
            if centre_error < 0.3:
                delta = float(np.clip(area - self.prev_area, -0.05, 0.05))
                reward += delta * 30.0
            else:
                # Penalise driving toward a badly off-centre box
                if action in (2, 3, 4):
                    reward -= 0.3

            # Spinning in place while the box is visible wastes time
            if action in (0, 1):
                reward -= 0.5

        # ── Time pressure ────────────────────────────────────────────────────
        reward -= 0.1
        return reward, False

    def close(self):
        p.disconnect(self.client)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ACTOR-CRITIC NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Shared trunk with separate policy (actor) and value (critic) heads.
    Actor outputs logits over 5 discrete actions.
    Critic outputs a scalar state value.
    """
    def __init__(self, state_dim=5, n_actions=5, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.trunk(x)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(self, state):
        """Sample an action; return (action, log_prob, value, entropy)."""
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, value, entropy

    def evaluate(self, states, actions):
        """Re-evaluate log-probs and values for stored (state, action) pairs."""
        logits, values = self.forward(states)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ROLLOUT BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores one on-policy rollout and computes GAE advantages."""

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, device):
        """
        GAE-Lambda advantage estimation.
        last_value: V(s_T) bootstrap value (0 if terminal).
        Returns tensors ready for PPO update.
        """
        n         = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        values_np  = np.array([v.item() for v in self.values] + [last_value],
                               dtype=np.float32)

        for t in reversed(range(n)):
            delta  = (self.rewards[t]
                      + GAMMA * values_np[t + 1] * (1 - self.dones[t])
                      - values_np[t])
            gae    = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values_np[:n]

        states    = torch.FloatTensor(np.array(self.states)).to(device)
        actions   = torch.LongTensor(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).to(device).detach()
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t    = torch.FloatTensor(returns).to(device)

        # Normalise advantages for training stability
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return states, actions, log_probs, advantages_t, returns_t

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PPO UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

def ppo_update(policy, optimizer, states, actions, old_log_probs,
               advantages, returns, device):
    """
    Run N_EPOCHS epochs of mini-batch PPO updates on the collected rollout.
    """
    n = len(states)
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0

    for _ in range(N_EPOCHS):
        # Shuffle indices for mini-batch sampling
        indices = torch.randperm(n)

        for start in range(0, n, MINI_BATCH_SIZE):
            mb_idx = indices[start: start + MINI_BATCH_SIZE]

            mb_states     = states[mb_idx]
            mb_actions    = actions[mb_idx]
            mb_old_lp     = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns    = returns[mb_idx]

            new_log_probs, values, entropy = policy.evaluate(mb_states, mb_actions)

            # Probability ratio r_t(θ)
            ratio = torch.exp(new_log_probs - mb_old_lp)

            # Clipped surrogate loss
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (MSE)
            value_loss = nn.functional.mse_loss(values, mb_returns)

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += (-entropy_loss.item())

    n_updates = N_EPOCHS * max(1, n // MINI_BATCH_SIZE)
    return (total_policy_loss / n_updates,
            total_value_loss  / n_updates,
            total_entropy     / n_updates)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    env    = HuskyEnv(gui=False)
    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    buffer    = RolloutBuffer()

    episode_rewards = []
    wins            = []
    best_avg        = -float("inf")

    # Running counters across rollout boundaries
    ep_reward   = 0.0
    ep_steps    = 0
    ep_won      = False
    ep_count    = 0
    ep_max_area = 0.0          # largest box area seen this episode
    total_steps = 0            # global env steps

    # Last PPO update stats (shown after each rollout)
    last_pl  = 0.0
    last_vl  = 0.0
    last_ent = 0.0

    state = env.reset()

    print(f"Starting PPO training (rollout={ROLLOUT_STEPS}, "
          f"epochs={N_EPOCHS}, mini_batch={MINI_BATCH_SIZE})")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'MaxArea':>7} {'BestAvg':>8}  Result")

    while ep_count < MAX_EPISODES:

        # ── Collect one rollout ──────────────────────────────────────────────
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
            ep_max_area  = max(ep_max_area, float(state[2]) if state[3] else ep_max_area)

            if done or ep_steps >= MAX_STEPS:
                ep_won   = done
                ep_count += 1
                episode_rewards.append(ep_reward)
                wins.append(int(ep_won))

                avg      = np.mean(episode_rewards[-20:])
                win_rate = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)

                if avg > best_avg:
                    best_avg = avg
                    torch.save(policy.state_dict(), "husky_ppo_best.pth")

                result = "WIN" if ep_won else "   "
                print(f"{ep_count:4d} {ep_reward:8.2f} {avg:8.2f} {win_rate*100:6.1f} "
                      f"{ep_steps:5d} {ep_max_area:7.3f} {best_avg:8.2f}  {result}")

                if ep_count % 50 == 0:
                    torch.save(policy.state_dict(), f"husky_ppo_ep{ep_count}.pth")
                    print(f"  → checkpoint: husky_ppo_ep{ep_count}.pth"
                          f"  [π {last_pl:.4f}  V {last_vl:.4f}  H {last_ent:.3f}"
                          f"  steps {total_steps}]")

                if len(wins) >= 50 and win_rate >= EARLY_STOP_WIN_RATE:
                    print(f"\nEarly stop: win rate {win_rate*100:.1f}% "
                          f"over last 50 eps.")
                    buffer.clear()
                    env.close()
                    torch.save(policy.state_dict(), "husky_ppo_final.pth")
                    print("Training complete. Models saved: "
                          "husky_ppo_final.pth, husky_ppo_best.pth")
                    return policy

                if ep_count >= MAX_EPISODES:
                    break

                # Reset for next episode
                state       = env.reset()
                ep_reward   = 0.0
                ep_steps    = 0
                ep_won      = False
                ep_max_area = 0.0

        # ── Bootstrap value at the end of the rollout ────────────────────────
        with torch.no_grad():
            s_t         = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, last_val, _, _ = policy.get_action(s_t)
            # If the last transition was terminal, bootstrap = 0
            last_value = 0.0 if (len(buffer.dones) > 0 and buffer.dones[-1] == 1.0) \
                              else last_val.item()

        # ── PPO update ───────────────────────────────────────────────────────
        states, actions, old_log_probs, advantages, returns = \
            buffer.compute_returns_and_advantages(last_value, device)

        last_pl, last_vl, last_ent = ppo_update(
            policy, optimizer,
            states, actions, old_log_probs, advantages, returns,
            device,
        )
        buffer.clear()
        print(f"  ↻ rollout update | π_loss {last_pl:.4f}"
              f"  V_loss {last_vl:.4f}"
              f"  entropy {last_ent:.3f}"
              f"  total_steps {total_steps}")

    env.close()
    torch.save(policy.state_dict(), "husky_ppo_final.pth")
    print("Training complete. Models saved: husky_ppo_final.pth, husky_ppo_best.pth")
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_trained(model_path="husky_ppo_best.pth", n_episodes=10):
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
                action = logits.argmax(dim=-1).item()  # greedy at test time

            state, reward, done = env.step(action)
            total_r += reward
            cx, _, area, vis, _ = state

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
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        path = sys.argv[2] if len(sys.argv) > 2 else "husky_ppo_best.pth"
        run_trained(model_path=path)
    else:
        train()
