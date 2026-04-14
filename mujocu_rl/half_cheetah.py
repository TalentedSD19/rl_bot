"""
HalfCheetah-v5: Comparing SAC, PPO, DDPG, TD3
============================================================
Requirements:
    pip install stable-baselines3[extra] gymnasium[mujoco] matplotlib numpy

Run:
    python train_halfcheetah.py

Outputs:
    - halfcheetah_comparison.png  → reward curves
    - models/sac_halfcheetah.zip  → trained SAC model
    - models/ppo_halfcheetah.zip  → trained PPO model
    - models/ddpg_halfcheetah.zip → trained DDPG model
    - models/td3_halfcheetah.zip  → trained TD3 model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID        = "HalfCheetah-v5"
TOTAL_STEPS   = 1_000_000        # steps per algorithm (reduce to 300_000 for a quick test)
EVAL_FREQ     = 10_000           # evaluate every N steps
N_EVAL_EPS    = 5                # episodes per evaluation
SEED          = 42
MODEL_DIR     = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Reward-tracking callback ───────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    """Evaluates the policy every `eval_freq` steps and logs mean reward."""

    def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps      = []
        self.mean_rewards   = []
        self.std_rewards    = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_r)
            self.std_rewards.append(std_r)
            if self.verbose:
                print(f"  [step {self.num_timesteps:>8,}] mean_reward={mean_r:.1f} ± {std_r:.1f}")
        return True


# ── Environment factory ────────────────────────────────────────────────────────
def make_env(seed=0):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(seed=0, normalize=False):
    env = DummyVecEnv([make_env(seed)])
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def make_eval_env(seed=99):
    """Single non-normalized env for fair evaluation."""
    return Monitor(gym.make(ENV_ID))


# ── Algorithm configurations ───────────────────────────────────────────────────
def get_action_noise(env):
    n_actions = env.action_space.shape[-1]
    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )


ALGOS = {
    "SAC": {
        "cls": SAC,
        "kwargs": dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=0,
            seed=SEED,
        ),
        "normalize": False,
    },
    "PPO": {
        "cls": PPO,
        "kwargs": dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=SEED,
        ),
        "normalize": True,   # PPO benefits from obs normalization
    },
    "DDPG": {
        "cls": DDPG,
        "kwargs": dict(
            policy="MlpPolicy",
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=0,
            seed=SEED,
        ),
        "normalize": False,
    },
    "TD3": {
        "cls": TD3,
        "kwargs": dict(
            policy="MlpPolicy",
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=0,
            seed=SEED,
        ),
        "normalize": False,
    },
}


# ── Training loop ──────────────────────────────────────────────────────────────
def train_all():
    results = {}

    for name, cfg in ALGOS.items():
        print(f"\n{'='*60}")
        print(f"  Training {name} on {ENV_ID}  [{TOTAL_STEPS:,} steps]")
        print(f"{'='*60}")

        train_env = make_vec_env(seed=SEED, normalize=cfg["normalize"])
        eval_env  = make_eval_env(seed=99)

        kwargs = cfg["kwargs"].copy()

        # Off-policy algorithms need action noise
        if name in ("DDPG", "TD3"):
            kwargs["action_noise"] = get_action_noise(train_env)

        model = cfg["cls"](env=train_env, **kwargs)

        callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPS,
            verbose=1,
        )

        model.learn(total_timesteps=TOTAL_STEPS, callback=callback, progress_bar=True)

        # Save model
        save_path = os.path.join(MODEL_DIR, f"{name.lower()}_halfcheetah")
        model.save(save_path)
        print(f"  ✓ Model saved → {save_path}.zip")

        # Final evaluation
        final_mean, final_std = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"  ✓ Final eval  → {final_mean:.1f} ± {final_std:.1f}")

        results[name] = {
            "timesteps"   : callback.timesteps,
            "mean_rewards": callback.mean_rewards,
            "std_rewards" : callback.std_rewards,
            "final_mean"  : final_mean,
            "final_std"   : final_std,
        }

        train_env.close()
        eval_env.close()

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────
PALETTE = {
    "SAC" : "#E63946",
    "PPO" : "#2196F3",
    "DDPG": "#FF9800",
    "TD3" : "#4CAF50",
}

def plot_comparison(results, save_path="halfcheetah_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1117")

    # ── Left: learning curves ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.set_title("Learning Curves — HalfCheetah-v5",
                 color="white", fontsize=14, fontweight="bold", pad=12)

    for name, data in results.items():
        ts   = np.array(data["timesteps"]) / 1e6   # millions
        mean = np.array(data["mean_rewards"])
        std  = np.array(data["std_rewards"])
        color = PALETTE[name]

        ax.plot(ts, mean, label=name, color=color, linewidth=2.0)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Timesteps (M)", color="#8b949e", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", color="#8b949e", fontsize=11)
    ax.tick_params(colors="#8b949e")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fM"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linestyle="--", linewidth=0.6)
    ax.legend(fontsize=11, facecolor="#161b22", labelcolor="white",
              edgecolor="#30363d", framealpha=0.9)

    # ── Right: final performance bar chart ────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.set_title("Final Performance (10-episode eval)",
                  color="white", fontsize=14, fontweight="bold", pad=12)

    names  = list(results.keys())
    finals = [results[n]["final_mean"] for n in names]
    errs   = [results[n]["final_std"]  for n in names]
    colors = [PALETTE[n] for n in names]
    x      = np.arange(len(names))

    bars = ax2.bar(x, finals, yerr=errs, color=colors, width=0.5,
                   error_kw=dict(ecolor="#8b949e", capsize=6, linewidth=1.5),
                   zorder=3)

    # Annotate bars
    for bar, val in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(errs) * 0.05,
                 f"{val:.0f}", ha="center", va="bottom",
                 color="white", fontsize=11, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color="#8b949e", fontsize=12)
    ax2.set_ylabel("Mean Episode Reward", color="#8b949e", fontsize=11)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.grid(axis="y", color="#21262d", linestyle="--", linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved → {save_path}")
    plt.show()


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "="*55)
    print(f"{'Algorithm':<10} {'Final Reward':>14} {'Std':>10} {'Best Step':>12}")
    print("-"*55)
    for name, data in results.items():
        idx  = int(np.argmax(data["mean_rewards"]))
        best = data["mean_rewards"][idx]
        step = data["timesteps"][idx]
        print(f"{name:<10} {data['final_mean']:>14.1f} "
              f"{data['final_std']:>10.1f} "
              f"{step:>10,} ({best:.0f})")
    print("="*55)


# ── Inference demo (optional) ──────────────────────────────────────────────────
def demo_best(results):
    """Loads the best algorithm's saved model and runs one episode."""
    best_algo = max(results, key=lambda n: results[n]["final_mean"])
    print(f"\n▶  Running demo with best algorithm: {best_algo}")

    cls       = ALGOS[best_algo]["cls"]
    save_path = os.path.join(MODEL_DIR, f"{best_algo.lower()}_halfcheetah")
    model     = cls.load(save_path)

    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    print(f"  Demo episode reward: {total_reward:.1f}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train_all()
    print_summary(results)
    plot_comparison(results)

    # Uncomment to watch the best agent play (requires a display):
    # demo_best(results)