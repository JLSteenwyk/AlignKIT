"""MSA-specific visualization: learning curves, tool selection, per-set comparison."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_msa import MSAConfig


def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL log file."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_msa_learning_curves(config: MSAConfig):
    """Plot smoothed reward, SP, TC over MSA training episodes."""
    train_log = _load_jsonl(config.log_dir / "msa_train.jsonl")
    if not train_log:
        print("No MSA training log found.")
        return

    steps = [e["step"] for e in train_log]
    rewards = [e.get("reward_mean", 0) for e in train_log]
    sps = [e.get("sp_mean", 0) for e in train_log]
    tcs = [e.get("tc_mean", 0) for e in train_log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, values, label, color in [
        (axes[0], rewards, "Reward", "blue"),
        (axes[1], sps, "SP Score", "green"),
        (axes[2], tcs, "TC Score", "orange"),
    ]:
        ax.plot(steps, values, alpha=0.3, color=color)
        if len(values) > 5:
            w = min(10, len(values) // 2)
            smoothed = _smooth(values, w)
            smooth_steps = steps[len(steps) - len(smoothed):]
            ax.plot(smooth_steps, smoothed, color=color, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(f"MSA Training {label}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.plot_dir / "msa_learning_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_msa_tool_selection(config: MSAConfig):
    """Plot MAFFT vs MUSCLE vs ClustalO selection fraction over training."""
    train_log = _load_jsonl(config.log_dir / "msa_train.jsonl")
    if not train_log:
        return

    steps = [e["step"] for e in train_log]
    mafft_fracs = [e.get("mafft_frac", 0) for e in train_log]
    muscle_fracs = [e.get("muscle_frac", 0) for e in train_log]
    clustalo_fracs = [e.get("clustalo_frac", 0) for e in train_log]

    fig, ax = plt.subplots(figsize=(10, 4))
    # Stacked area: MAFFT at bottom, then MUSCLE, then ClustalO
    mafft_top = mafft_fracs
    muscle_top = [m + u for m, u in zip(mafft_fracs, muscle_fracs)]
    clustalo_top = [mu + c for mu, c in zip(muscle_top, clustalo_fracs)]

    ax.fill_between(steps, 0, mafft_top, alpha=0.4, color="steelblue", label="MAFFT")
    ax.fill_between(steps, mafft_top, muscle_top, alpha=0.4, color="coral", label="MUSCLE")
    ax.fill_between(steps, muscle_top, clustalo_top, alpha=0.4, color="seagreen", label="ClustalO")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction")
    ax.set_title("Tool Selection Over Training")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.plot_dir / "msa_tool_selection.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_msa_per_set_comparison(config: MSAConfig):
    """Plot per-reference-set grouped bar chart: agent vs MAFFT default vs MUSCLE default."""
    eval_path = config.log_dir / "msa_eval_results.json"
    if not eval_path.exists():
        print("No MSA evaluation results found.")
        return

    with open(eval_path) as f:
        eval_results = json.load(f)

    per_set = eval_results.get("per_set", {})
    if not per_set:
        return

    ref_sets = sorted(per_set.keys())
    agent_rewards = [per_set[rs]["reward_mean"] for rs in ref_sets]
    agent_sp = [per_set[rs]["sp_mean"] for rs in ref_sets]
    agent_tc = [per_set[rs]["tc_mean"] for rs in ref_sets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(ref_sets))
    width = 0.6

    for ax, values, title in [
        (axes[0], agent_rewards, "Reward"),
        (axes[1], agent_sp, "SP Score"),
        (axes[2], agent_tc, "TC Score"),
    ]:
        ax.bar(x, values, width, color="steelblue", label="RL Agent")
        ax.set_xlabel("Reference Set")
        ax.set_ylabel(title)
        ax.set_title(f"MSA {title} by Set")
        ax.set_xticks(x)
        ax.set_xticklabels(ref_sets)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = config.plot_dir / "msa_per_set_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_msa_action_distribution(config: MSAConfig):
    """Plot distribution of tool+strategy configurations chosen by the agent."""
    eval_path = config.log_dir / "msa_eval_results.json"
    if not eval_path.exists():
        return

    with open(eval_path) as f:
        eval_results = json.load(f)

    tool_dist = eval_results.get("tool_distribution", {})
    if not tool_dist:
        return

    labels = list(tool_dist.keys())
    sizes = list(tool_dist.values())
    colors = ["steelblue", "coral", "seagreen", "gray"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Tool Selection Distribution")

    plt.tight_layout()
    save_path = config.plot_dir / "msa_tool_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def generate_msa_plots(config: MSAConfig):
    """Generate all MSA visualization plots."""
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating MSA plots...")
    plot_msa_learning_curves(config)
    plot_msa_tool_selection(config)
    plot_msa_per_set_comparison(config)
    plot_msa_action_distribution(config)
    print("All MSA plots generated.")
