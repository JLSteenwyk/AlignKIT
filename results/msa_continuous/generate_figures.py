"""Generate figures for the continuous MSA RL parameter learning results.

Run:
    python results/msa_continuous/generate_figures.py

Produces Fig 1-4 as PDF and PNG in this directory.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs_msa")

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, "%s.%s" % (name, ext))
        fig.savefig(path)
        print("  saved %s" % path)
    plt.close(fig)


def load_train_log():
    records = []
    path = os.path.join(LOG_DIR, "msa_train.jsonl")
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_eval_log():
    records = []
    path = os.path.join(LOG_DIR, "msa_eval.jsonl")
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def smooth(values, window=20):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Fig 1: Training Curves (Reward, SP, TC)
# ---------------------------------------------------------------------------

def fig1(train_log, eval_log):
    """3-panel training curves: reward, SP, TC with eval overlay."""
    episodes = [r["episode"] for r in train_log]
    rewards = [r["reward_mean"] for r in train_log]
    sps = [r["sp_mean"] for r in train_log]
    tcs = [r["tc_mean"] for r in train_log]

    eval_eps = [r["episode"] for r in eval_log]
    eval_rewards = [r["reward_mean"] for r in eval_log]
    eval_sps = [r["sp_mean"] for r in eval_log]
    eval_tcs = [r["tc_mean"] for r in eval_log]

    window = 20
    sm_eps = episodes[window - 1:]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharex=True)

    metrics = [
        ("Reward", rewards, eval_rewards),
        ("SP Score", sps, eval_sps),
        ("TC Score", tcs, eval_tcs),
    ]

    for ax, (title, train_vals, eval_vals) in zip(axes, metrics):
        # Raw training (light)
        ax.plot(episodes, train_vals, alpha=0.15, color=BLUE, linewidth=0.5)
        # Smoothed training
        ax.plot(sm_eps, smooth(train_vals, window), color=BLUE, linewidth=1.5,
                label="Train (smoothed)")
        # Eval points
        ax.plot(eval_eps, eval_vals, "o-", color=ORANGE, markersize=3,
                linewidth=1.2, label="Eval (greedy)")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        if ax == axes[0]:
            ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

    axes[0].set_ylabel("Score")
    fig.suptitle("Fig 1: MSA Training Curves", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_training_curves")


# ---------------------------------------------------------------------------
# Fig 2: Learned Parameter Trajectory (op, ep)
# ---------------------------------------------------------------------------

def fig2(train_log):
    """2-panel: learned op and ep over training."""
    episodes = [r["episode"] for r in train_log]
    ops = [r["op_mean"] for r in train_log]
    eps_vals = [r["ep_mean"] for r in train_log]

    window = 20
    sm_eps = episodes[window - 1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0), sharex=True)

    # op
    ax1.plot(episodes, ops, alpha=0.2, color=BLUE, linewidth=0.5)
    ax1.plot(sm_eps, smooth(ops, window), color=BLUE, linewidth=1.5, label="Learned op")
    ax1.axhline(1.53, color=RED, linestyle="--", linewidth=1.0, label="Default (1.53)")
    ax1.set_ylabel("Opening penalty (op)")
    ax1.set_xlabel("Episode")
    ax1.set_title("Gap Opening Penalty")
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax1.set_ylim(0.0, 5.5)

    # ep
    ax2.plot(episodes, eps_vals, alpha=0.2, color=GREEN, linewidth=0.5)
    ax2.plot(sm_eps, smooth(eps_vals, window), color=GREEN, linewidth=1.5, label="Learned ep")
    ax2.axhline(0.0, color=RED, linestyle="--", linewidth=1.0, label="Default (0.0)")
    ax2.set_ylabel("Extension penalty (ep)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Gap Extension Penalty")
    ax2.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax2.set_ylim(-0.1, 1.1)

    fig.suptitle("Fig 2: Learned MAFFT Parameters Over Training", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_parameter_trajectory")


# ---------------------------------------------------------------------------
# Fig 3: Baseline Comparison Bar Chart
# ---------------------------------------------------------------------------

def fig3(eval_log):
    """Bar chart: MAFFT --auto vs L-INS-i default vs RL Agent."""
    # Baselines (computed earlier in the session)
    methods = ["MAFFT\n--auto", "MAFFT L-INS-i\ndefault", "RL Agent\n(greedy)"]

    # Values from our baseline evaluation
    reward_vals = [0.4455, 0.4483, 0.4543]
    sp_vals = [0.6981, 0.7025, 0.7053]
    tc_vals = [0.1929, 0.1941, 0.2033]

    # Use last eval record for RL agent (most recent)
    if eval_log:
        last_eval = eval_log[-1]
        reward_vals[2] = last_eval["reward_mean"]
        sp_vals[2] = last_eval["sp_mean"]
        tc_vals[2] = last_eval["tc_mean"]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bars1 = ax.bar(x - width, reward_vals, width, label="Reward", color=BLUE,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, sp_vals, width, label="SP", color=GREEN,
                   edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, tc_vals, width, label="TC", color=ORANGE,
                   edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    "%.3f" % h, ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")
    ax.set_title("Fig 3: Baseline Comparison (RV40 + RV50, n=65)")

    # Add improvement annotations
    # RL vs auto
    auto_r = reward_vals[0]
    rl_r = reward_vals[2]
    pct = 100 * (rl_r - auto_r) / auto_r
    ax.annotate("+%.1f%%" % pct, xy=(2 - width, rl_r + 0.02),
                fontsize=8, color=RED, fontweight="bold", ha="center")

    fig.tight_layout()
    save_fig(fig, "fig3_baseline_comparison")


# ---------------------------------------------------------------------------
# Fig 4: Per-Dataset Breakdown
# ---------------------------------------------------------------------------

def fig4(eval_log):
    """Grouped bar chart: RV40 vs RV50 for each method."""
    # Baseline per-set values from our evaluation
    data = {
        "MAFFT --auto": {
            "RV40": {"reward": 0.4375, "sp": 0.6947, "tc": 0.1803},
            "RV50": {"reward": 0.4698, "sp": 0.7083, "tc": 0.2314},
        },
        "L-INS-i default": {
            "RV40": {"reward": 0.4412, "sp": 0.7006, "tc": 0.1819},
            "RV50": {"reward": 0.4698, "sp": 0.7083, "tc": 0.2314},
        },
        "RL Agent": {
            "RV40": {"reward": 0.4503, "sp": 0.7065, "tc": 0.1942},
            "RV50": {"reward": 0.4665, "sp": 0.7018, "tc": 0.2311},
        },
    }

    # Update RL agent from latest eval log
    if eval_log:
        last_eval = eval_log[-1]
        per_set = last_eval.get("per_set", {})
        if "RV40" in per_set:
            data["RL Agent"]["RV40"] = {
                "reward": per_set["RV40"]["reward_mean"],
                "sp": per_set["RV40"]["sp_mean"],
                "tc": per_set["RV40"]["tc_mean"],
            }
        if "RV50" in per_set:
            data["RL Agent"]["RV50"] = {
                "reward": per_set["RV50"]["reward_mean"],
                "sp": per_set["RV50"]["sp_mean"],
                "tc": per_set["RV50"]["tc_mean"],
            }

    methods = list(data.keys())
    datasets = ["RV40", "RV50"]
    colors = [BLUE, ORANGE, GREEN]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    metric_names = [("reward", "Reward"), ("sp", "SP Score"), ("tc", "TC Score")]

    for ax, (metric_key, metric_label) in zip(axes, metric_names):
        x = np.arange(len(datasets))
        width = 0.22

        for i, method in enumerate(methods):
            vals = [data[method][ds][metric_key] for ds in datasets]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=method,
                         color=colors[i], edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        "%.3f" % h, ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        if ax == axes[0]:
            ax.legend(frameon=True, fancybox=False, edgecolor="0.8", fontsize=7)

    fig.suptitle("Fig 4: Per-Dataset Breakdown", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig4_per_dataset")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading training log...")
    train_log = load_train_log()
    print("  %d batches, %d episodes" % (len(train_log), train_log[-1]["episode"]))

    print("Loading eval log...")
    eval_log = load_eval_log()
    print("  %d eval checkpoints" % len(eval_log))

    print("\nGenerating Fig 1: Training Curves")
    fig1(train_log, eval_log)

    print("\nGenerating Fig 2: Parameter Trajectory")
    fig2(train_log)

    print("\nGenerating Fig 3: Baseline Comparison")
    fig3(eval_log)

    print("\nGenerating Fig 4: Per-Dataset Breakdown")
    fig4(eval_log)

    print("\nDone -- all figures saved to %s" % OUT_DIR)


if __name__ == "__main__":
    main()
