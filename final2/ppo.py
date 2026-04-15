"""
ppo.py  --  Proximal Policy Optimisation

  RolloutBuffer   : collects transitions, computes GAE
  ppo_update      : one PPO update cycle over the buffer
  train           : full training loop
  run_trained     : load checkpoint and visualise
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

from config import PPO, MAX_STEPS, PHASE_NAMES, ACTION_NAMES
from environment import HuskyTask2Env
from models import ActorCritic, apply_spin_mask

_C = PPO   # shorthand


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

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

    def compute_gae(self, last_value: float, device):
        """Returns (states, actions, log_probs, advantages, returns, old_values) as tensors."""
        n          = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        vals       = np.array([v.item() for v in self.values] + [last_value],
                               dtype=np.float32)

        for t in reversed(range(n)):
            delta         = (self.rewards[t]
                             + _C["GAMMA"] * vals[t + 1] * (1 - self.dones[t])
                             - vals[t])
            gae           = delta + _C["GAMMA"] * _C["GAE_LAMBDA"] * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns    = advantages + vals[:n]
        states     = torch.FloatTensor(np.array(self.states)).to(device)
        actions    = torch.LongTensor(self.actions).to(device)
        log_probs  = torch.stack(self.log_probs).to(device).detach()
        adv_t      = torch.FloatTensor(advantages).to(device)
        ret_t      = torch.FloatTensor(returns).to(device)
        old_vals_t = torch.FloatTensor(vals[:n]).to(device)
        adv_t      = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return states, actions, log_probs, adv_t, ret_t, old_vals_t

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy, optimizer, states, actions, old_log_probs,
               advantages, returns, old_values):
    n = len(states)
    pl_sum = vl_sum = ent_sum = 0.0
    n_updates = 0
    clip = _C["CLIP_EPS"]

    for _ in range(_C["N_EPOCHS"]):
        idx        = torch.randperm(n)
        kl_epoch   = 0.0
        n_batches  = 0

        for start in range(0, n, _C["MINI_BATCH"]):
            mb = idx[start: start + _C["MINI_BATCH"]]
            new_lp, vals, entropy = policy.evaluate(states[mb], actions[mb])

            logratio = new_lp - old_log_probs[mb]
            ratio    = torch.exp(logratio)
            adv      = advantages[mb]

            # Policy loss (clipped surrogate)
            p_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            ).mean()

            # Value loss with clipping
            v_unclipped = (vals - returns[mb]).pow(2)
            v_clipped   = old_values[mb] + torch.clamp(
                vals - old_values[mb], -clip, clip)
            v_loss      = 0.5 * torch.max(
                v_unclipped, (v_clipped - returns[mb]).pow(2)).mean()

            loss = p_loss + _C["VALUE_COEF"] * v_loss - _C["ENTROPY_COEF"] * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), _C["GRAD_CLIP"])
            optimizer.step()

            pl_sum  += p_loss.item()
            vl_sum  += v_loss.item()
            ent_sum += entropy.mean().item()
            n_updates += 1

            # Approximate KL for early stopping (no grad needed)
            with torch.no_grad():
                kl_epoch += ((ratio - 1) - logratio).mean().item()
            n_batches += 1

        # Abort remaining epochs if policy has drifted too far
        if n_batches > 0 and (kl_epoch / n_batches) > _C["TARGET_KL"]:
            break

    return (pl_sum / max(1, n_updates),
            vl_sum / max(1, n_updates),
            ent_sum / max(1, n_updates))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(save_prefix: str = "husky_ppo"):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env       = HuskyTask2Env(gui=False)
    policy    = ActorCritic(hidden=_C["HIDDEN"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=_C["LR"], eps=1e-5)
    # Linearly decay LR to 10 % of its initial value over all episodes
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                         total_iters=_C["MAX_EPISODES"])
    buffer    = RolloutBuffer()

    ep_rewards  = []
    wins        = []
    best_avg    = -float("inf")
    ep_reward   = 0.0
    ep_steps    = 0
    ep_won      = False
    ep_count    = 0
    total_steps = 0
    last_pl = last_vl = last_ent = 0.0

    state = env.reset()

    print(f"\nStarting  rollout={_C['ROLLOUT_STEPS']}  "
          f"epochs={_C['N_EPOCHS']}  batch={_C['MINI_BATCH']}")
    print(f"{'Ep':>4} {'Reward':>8} {'Avg20':>8} {'Win%50':>6} "
          f"{'Steps':>5} {'Phase':>7} {'BestAvg':>8}  Result")

    while ep_count < _C["MAX_EPISODES"]:

        for _ in range(_C["ROLLOUT_STEPS"]):
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
                win_rate = (np.mean(wins[-50:]) if len(wins) >= 50
                            else np.mean(wins))
                phase_tag = PHASE_NAMES[env.phase]

                if avg > best_avg:
                    best_avg = avg
                    torch.save(policy.state_dict(), f"{save_prefix}_best.pth")

                tag = "WIN" if ep_won else "   "
                print(f"{ep_count:4d} {ep_reward:8.2f} {avg:8.2f} "
                      f"{win_rate*100:6.1f} {ep_steps:5d} {phase_tag:>7} "
                      f"{best_avg:8.2f}  {tag}")

                if ep_count % 100 == 0:
                    ckpt = f"{save_prefix}_ep{ep_count}.pth"
                    torch.save(policy.state_dict(), ckpt)
                    print(f"  ↳ checkpoint {ckpt}  "
                          f"[π {last_pl:.4f}  V {last_vl:.4f}  "
                          f"H {last_ent:.3f}  steps {total_steps}]")

                if len(wins) >= 50 and win_rate >= _C["EARLY_STOP_RATE"]:
                    print(f"\nEarly stop: win rate {win_rate*100:.1f}% over last 50 eps.")
                    env.close()
                    torch.save(policy.state_dict(), f"{save_prefix}_final.pth")
                    print(f"Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
                    return policy

                if ep_count >= _C["MAX_EPISODES"]:
                    break

                state = env.reset()
                ep_reward, ep_steps, ep_won = 0.0, 0, False

        # Bootstrap + PPO update
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, last_val = policy(s_t)
            last_value = 0.0 if buffer.dones[-1] == 1.0 else last_val.item()

        states, actions, old_lp, adv, ret, old_vals = buffer.compute_gae(last_value, device)
        last_pl, last_vl, last_ent = ppo_update(
            policy, optimizer, states, actions, old_lp, adv, ret, old_vals)
        scheduler.step()
        buffer.clear()
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  ↻ update | π {last_pl:.4f}  V {last_vl:.4f}  "
              f"H {last_ent:.3f}  lr {cur_lr:.2e}  steps {total_steps}")

    env.close()
    torch.save(policy.state_dict(), f"{save_prefix}_final.pth")
    print(f"Done. Saved: {save_prefix}_final.pth  {save_prefix}_best.pth")
    return policy


# ---------------------------------------------------------------------------
# Inference / visualisation
# ---------------------------------------------------------------------------

def run_trained(model_path: str = "husky_ppo_best.pth", n_episodes: int = 5):
    device = torch.device("cpu")
    policy = ActorCritic(hidden=_C["HIDDEN"]).to(device)
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
                logits, _ = policy(s_t)
                logits    = apply_spin_mask(logits, s_t)
                action    = logits.argmax(dim=-1).item()

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
