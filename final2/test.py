"""
test.py -- Performance comparison of PPO, SAC, and TQC on HuskyTask2Env.

Runs N evaluation episodes per algorithm (no training, no GUI),
collects detailed metrics, prints a summary table, and saves a
multi-panel comparison figure.

Usage
-----
    python test.py                   # 20 episodes, saves comparison.png
    python test.py -n 50             # 50 episodes
    python test.py -n 30 -o out.png  # custom output filename
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                 # save to file, no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PPO, SAC, TQC, MAX_STEPS, N_ACTIONS, PHASE_NAMES, ACTION_NAMES
from environment import HuskyTask2Env
from models import ActorCritic, SACDiscreteActor, apply_spin_mask

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS = {
    "PPO": os.path.join(SCRIPT_DIR, "husky_ppo_best.pth"),
    "SAC": os.path.join(SCRIPT_DIR, "husky_sac_best.pth"),
    "TQC": os.path.join(SCRIPT_DIR, "husky_tqc_best.pth"),
}

ALGO_COLORS = {
    "PPO": "#4C72B0",   # blue
    "SAC": "#DD8452",   # orange
    "TQC": "#55A868",   # green
}

# ── Model loaders ─────────────────────────────────────────────────────────────

def load_ppo(path: str) -> ActorCritic:
    model = ActorCritic(hidden=PPO["HIDDEN"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_sac(path: str) -> SACDiscreteActor:
    actor = SACDiscreteActor(hidden=SAC["HIDDEN"])
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def load_tqc(path: str) -> SACDiscreteActor:
    # TQC's actor is a SACDiscreteActor; the checkpoint is a dict with "actor" key
    actor = SACDiscreteActor(hidden=TQC["HIDDEN"])
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


# ── Greedy action selectors ───────────────────────────────────────────────────

def ppo_action(model: ActorCritic, state: np.ndarray) -> int:
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(s)
        logits    = apply_spin_mask(logits, s)
    return int(logits.argmax(dim=-1).item())


def sac_tqc_action(actor: SACDiscreteActor, state: np.ndarray) -> int:
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        probs, _, _ = actor.evaluate(s)
    return int(probs.argmax(1).item())


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, select_fn, n_episodes: int) -> dict:
    """
    Run n_episodes in DIRECT mode (headless) and collect:
      - per-episode total reward
      - steps taken
      - win / no-win
      - final task phase when the episode ended
      - action frequency counts
      - every individual step reward
    """
    env = HuskyTask2Env(gui=False)

    ep_rewards    = []
    ep_steps      = []
    ep_won        = []
    ep_final_phase = []
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    step_rewards  = []

    print(f"\n{'─' * 56}")
    print(f"  Evaluating  {name}  ({n_episodes} episodes)")
    print(f"{'─' * 56}")
    print(f"  {'Ep':>3}  {'Reward':>9}  {'Steps':>5}  {'Phase':>7}  Result")

    for ep in range(1, n_episodes + 1):
        state  = env.reset()
        ep_r   = 0.0
        won    = False
        last_step = MAX_STEPS - 1

        for step in range(MAX_STEPS):
            action = select_fn(state)
            action_counts[action] += 1

            next_state, reward, done = env.step(action)
            ep_r         += reward
            step_rewards.append(reward)
            state         = next_state

            if done:
                won       = True
                last_step = step
                break
        else:
            last_step = MAX_STEPS - 1

        ep_rewards.append(ep_r)
        ep_steps.append(last_step + 1)
        ep_won.append(int(won))
        ep_final_phase.append(env.phase)

        tag = "WIN" if won else "   "
        print(f"  {ep:3d}  {ep_r:9.2f}  {last_step+1:5d}  "
              f"{PHASE_NAMES[env.phase]:>7}  {tag}")

    env.close()

    wr   = np.mean(ep_won) * 100
    mu   = np.mean(ep_rewards)
    sd   = np.std(ep_rewards)
    print(f"\n  Win rate : {wr:.1f}%  |  "
          f"Reward : {mu:.2f} ± {sd:.2f}  |  "
          f"Avg steps : {np.mean(ep_steps):.1f}")

    return {
        "rewards":       np.array(ep_rewards),
        "steps":         np.array(ep_steps),
        "won":           np.array(ep_won),
        "final_phase":   np.array(ep_final_phase),
        "action_counts": action_counts,
        "step_rewards":  np.array(step_rewards),
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: dict, n_episodes: int):
    algos = list(results.keys())

    print(f"\n{'═' * 72}")
    print(f"  {'PERFORMANCE SUMMARY':^68}")
    print(f"  Evaluation episodes per algorithm: {n_episodes}")
    print(f"{'═' * 72}")
    header = f"  {'Metric':<32} " + "  ".join(f"{a:>10}" for a in algos)
    print(header)
    print(f"{'─' * 72}")

    def row(label, values):
        print(f"  {label:<32} " + "  ".join(f"{v:>10}" for v in values))

    def wins(a):
        won_idx = results[a]["won"] == 1
        return results[a]["steps"][won_idx] if won_idx.any() else np.array([np.nan])

    row("Win Rate (%)",
        [f"{np.mean(results[a]['won'])*100:.1f}" for a in algos])
    row("Mean Episode Reward",
        [f"{np.mean(results[a]['rewards']):.2f}" for a in algos])
    row("Std Episode Reward",
        [f"{np.std(results[a]['rewards']):.2f}"  for a in algos])
    row("Min Episode Reward",
        [f"{np.min(results[a]['rewards']):.2f}"  for a in algos])
    row("Max Episode Reward",
        [f"{np.max(results[a]['rewards']):.2f}"  for a in algos])
    row("Median Episode Reward",
        [f"{np.median(results[a]['rewards']):.2f}" for a in algos])
    print(f"{'─' * 72}")
    row("Avg Steps (all eps)",
        [f"{np.mean(results[a]['steps']):.1f}" for a in algos])
    row("Avg Steps (wins only)",
        [f"{np.nanmean(wins(a)):.1f}" for a in algos])
    row("Std Steps (wins only)",
        [f"{np.nanstd(wins(a)):.1f}"  for a in algos])
    print(f"{'─' * 72}")
    row("Mean Step Reward",
        [f"{np.mean(results[a]['step_rewards']):.4f}" for a in algos])
    row("Std Step Reward",
        [f"{np.std(results[a]['step_rewards']):.4f}"  for a in algos])
    print(f"{'═' * 72}")

    # Per-phase breakdown
    print(f"\n  Final Phase Distribution (% of episodes)")
    print(f"{'─' * 72}")
    phase_header = f"  {'Phase':<20} " + "  ".join(f"{a:>10}" for a in algos)
    print(phase_header)
    print(f"{'─' * 72}")
    for pi, pname in enumerate(PHASE_NAMES):
        vals = [f"{(results[a]['final_phase'] == pi).mean()*100:.1f}%" for a in algos]
        print(f"  {pname.strip():<20} " + "  ".join(f"{v:>10}" for v in vals))
    print(f"{'═' * 72}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, n_episodes: int, out_path: str):
    algos = list(results.keys())
    colors = [ALGO_COLORS[a] for a in algos]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Algorithm Comparison — PPO vs SAC vs TQC  "
        f"({n_episodes} evaluation episodes each)",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── Panel 1: Mean reward ± std (bar) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    means = [np.mean(results[a]["rewards"]) for a in algos]
    stds  = [np.std(results[a]["rewards"])  for a in algos]
    bars  = ax1.bar(algos, means, yerr=stds, capsize=7,
                    color=colors, alpha=0.85, width=0.5,
                    error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 m + s + max(abs(m) * 0.02, 1),
                 f"{m:.1f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax1.set_title("Mean Episode Reward ± Std", fontsize=11)
    ax1.set_ylabel("Total Reward")
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Win rate (bar) ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    win_rates = [np.mean(results[a]["won"]) * 100 for a in algos]
    bars2     = ax2.bar(algos, win_rates, color=colors, alpha=0.85, width=0.5)
    for bar, wr in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{wr:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_title("Win Rate", fontsize=11)
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Average steps per episode (bar) ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    s_means = [np.mean(results[a]["steps"]) for a in algos]
    s_stds  = [np.std(results[a]["steps"])  for a in algos]
    bars3   = ax3.bar(algos, s_means, yerr=s_stds, capsize=7,
                      color=colors, alpha=0.85, width=0.5,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, m in zip(bars3, s_means):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 8,
                 f"{m:.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax3.axhline(MAX_STEPS, color="red", linewidth=1.0, linestyle="--",
                label=f"Max ({MAX_STEPS})")
    ax3.set_title("Avg Steps per Episode", fontsize=11)
    ax3.set_ylabel("Steps")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ── Panel 4: Episode reward over time (line) ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    eps = range(1, n_episodes + 1)
    for a in algos:
        ax4.plot(eps, results[a]["rewards"],
                 label=a, color=ALGO_COLORS[a],
                 marker="o", markersize=4, linewidth=1.6, alpha=0.9)
        ax4.axhline(np.mean(results[a]["rewards"]),
                    color=ALGO_COLORS[a], linewidth=1.0,
                    linestyle="--", alpha=0.55)
    ax4.set_title("Episode Reward per Evaluation Episode", fontsize=11)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Total Reward")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ── Panel 5: Reward distribution (box plot) ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    bp  = ax5.boxplot(
        [results[a]["rewards"] for a in algos],
        labels=algos,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    for patch, a in zip(bp["boxes"], algos):
        patch.set_facecolor(ALGO_COLORS[a])
        patch.set_alpha(0.75)
    ax5.set_title("Reward Distribution", fontsize=11)
    ax5.set_ylabel("Total Reward")
    ax5.grid(axis="y", alpha=0.3)

    # ── Panel 6: Final phase distribution (stacked bar) ───────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    phase_pct = {}
    for a in algos:
        counts = np.zeros(len(PHASE_NAMES), dtype=float)
        for ph in results[a]["final_phase"]:
            counts[ph] += 1
        phase_pct[a] = counts / n_episodes * 100

    x_pos      = np.arange(len(algos))
    bar_width  = 0.55
    phase_cmap = plt.cm.tab10(np.linspace(0, 0.9, len(PHASE_NAMES)))
    bottom     = np.zeros(len(algos))

    for pi, pname in enumerate(PHASE_NAMES):
        vals   = [phase_pct[a][pi] for a in algos]
        brs    = ax6.bar(x_pos, vals, bar_width,
                         bottom=bottom, color=phase_cmap[pi],
                         label=pname.strip(), alpha=0.88)
        for bar, v in zip(brs, vals):
            if v > 5:
                ax6.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2,
                         f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(algos, fontsize=11)
    ax6.set_title("Final Phase Distribution (% of Episodes)", fontsize=11)
    ax6.set_ylabel("% of Episodes")
    ax6.set_ylim(0, 108)
    ax6.legend(loc="upper right", fontsize=8, ncol=2,
               framealpha=0.85, title="Phase", title_fontsize=8)

    # ── Panel 7: Action distribution (grouped bar) ────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    x_act = np.arange(N_ACTIONS)
    w     = 0.22
    for i, a in enumerate(algos):
        counts = results[a]["action_counts"]
        pct    = counts / counts.sum() * 100
        ax7.bar(x_act + i * w, pct, w,
                label=a, color=ALGO_COLORS[a], alpha=0.85)
    ax7.set_xticks(x_act + w)
    ax7.set_xticklabels([n.strip() for n in ACTION_NAMES],
                        rotation=22, ha="right", fontsize=8)
    ax7.set_title("Action Frequency (%)", fontsize=11)
    ax7.set_ylabel("% of All Steps")
    ax7.legend(fontsize=8)
    ax7.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved  →  {out_path}")
    plt.close(fig)


# ── Additional focused plots ──────────────────────────────────────────────────

def plot_steps_vs_reward(results: dict, n_episodes: int, out_path: str):
    """Scatter: steps vs reward per episode, coloured by algorithm."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for a, res in results.items():
        wins = res["won"].astype(bool)
        ax.scatter(res["steps"][wins],  res["rewards"][wins],
                   label=f"{a} (win)",  color=ALGO_COLORS[a],
                   marker="*", s=120, zorder=3)
        ax.scatter(res["steps"][~wins], res["rewards"][~wins],
                   label=f"{a} (loss)", color=ALGO_COLORS[a],
                   marker="x", s=60,  alpha=0.6)
    ax.set_xlabel("Steps Taken")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Steps vs Reward  (★ = win, ✕ = loss)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved  →  {out_path}")
    plt.close(fig)


def plot_cumulative_wins(results: dict, n_episodes: int, out_path: str):
    """Cumulative win count over evaluation episodes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    eps = range(1, n_episodes + 1)
    for a, res in results.items():
        cumulative = np.cumsum(res["won"])
        ax.plot(eps, cumulative, label=a, color=ALGO_COLORS[a],
                linewidth=2.2, marker="o", markersize=4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Wins")
    ax.set_title("Cumulative Wins across Evaluation Episodes")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Cumulative wins plot saved  →  {out_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare PPO, SAC, TQC on HuskyTask2Env")
    parser.add_argument("-n", "--episodes", type=int, default=20,
                        help="Evaluation episodes per algorithm (default: 20)")
    parser.add_argument("-o", "--output", default="comparison.png",
                        help="Main output plot filename (default: comparison.png)")
    args = parser.parse_args()

    # -- Verify checkpoints ---------------------------------------------------
    missing = [name for name, path in CHECKPOINTS.items()
               if not os.path.isfile(path)]
    if missing:
        print(f"ERROR: Missing checkpoint(s): {missing}")
        print("Make sure the following files exist:")
        for name in missing:
            print(f"  {CHECKPOINTS[name]}")
        sys.exit(1)

    # -- Load models ----------------------------------------------------------
    print("Loading trained models...")
    ppo_model = load_ppo(CHECKPOINTS["PPO"])
    sac_actor = load_sac(CHECKPOINTS["SAC"])
    tqc_actor = load_tqc(CHECKPOINTS["TQC"])
    print("  PPO, SAC, TQC  —  all loaded.")

    # -- Evaluate -------------------------------------------------------------
    results = {}
    results["PPO"] = evaluate("PPO",
                               lambda s: ppo_action(ppo_model, s),
                               args.episodes)
    results["SAC"] = evaluate("SAC",
                               lambda s: sac_tqc_action(sac_actor, s),
                               args.episodes)
    results["TQC"] = evaluate("TQC",
                               lambda s: sac_tqc_action(tqc_actor, s),
                               args.episodes)

    # -- Summary table --------------------------------------------------------
    print_summary_table(results, args.episodes)

    # -- Plots ----------------------------------------------------------------
    base = os.path.splitext(os.path.join(SCRIPT_DIR, args.output))[0]

    plot_comparison(results, args.episodes,
                    out_path=base + ".png")
    plot_steps_vs_reward(results, args.episodes,
                         out_path=base + "_scatter.png")
    plot_cumulative_wins(results, args.episodes,
                         out_path=base + "_cumwins.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
