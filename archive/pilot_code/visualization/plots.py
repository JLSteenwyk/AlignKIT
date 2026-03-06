"""Visualization: learning curves, action heatmaps, per-set bar charts."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from config import Config
from env.action_space import ActionSpace


def _smooth(values: List[float], window: int = 20) -> np.ndarray:
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


def plot_learning_curves(config: Config):
    """Plot smoothed reward, SP, TC over training episodes."""
    train_log = _load_jsonl(config.log_dir / "train.jsonl")
    if not train_log:
        print("No training log found.")
        return

    steps = [e["step"] for e in train_log]
    rewards = [e.get("reward_mean", 0) for e in train_log]
    sps = [e.get("sp_mean", 0) for e in train_log]
    tcs = [e.get("tc_mean", 0) for e in train_log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Reward
    axes[0].plot(steps, rewards, alpha=0.3, color="blue")
    if len(rewards) > 10:
        smoothed = _smooth(rewards, min(20, len(rewards) // 2))
        smooth_steps = steps[len(steps) - len(smoothed):]
        axes[0].plot(smooth_steps, smoothed, color="blue", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward")
    axes[0].grid(True, alpha=0.3)

    # SP
    axes[1].plot(steps, sps, alpha=0.3, color="green")
    if len(sps) > 10:
        smoothed = _smooth(sps, min(20, len(sps) // 2))
        smooth_steps = steps[len(steps) - len(smoothed):]
        axes[1].plot(smooth_steps, smoothed, color="green", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("SP Score")
    axes[1].set_title("SP Score")
    axes[1].grid(True, alpha=0.3)

    # TC
    axes[2].plot(steps, tcs, alpha=0.3, color="orange")
    if len(tcs) > 10:
        smoothed = _smooth(tcs, min(20, len(tcs) // 2))
        smooth_steps = steps[len(steps) - len(smoothed):]
        axes[2].plot(smooth_steps, smoothed, color="orange", linewidth=2)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("TC Score")
    axes[2].set_title("TC Score")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.plot_dir / "learning_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_entropy(config: Config):
    """Plot entropy decay over training."""
    train_log = _load_jsonl(config.log_dir / "train.jsonl")
    if not train_log:
        return

    steps = [e["step"] for e in train_log]
    entropy = [e.get("entropy", 0) for e in train_log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, entropy, color="red", alpha=0.5)
    if len(entropy) > 10:
        smoothed = _smooth(entropy, min(20, len(entropy) // 2))
        smooth_steps = steps[len(steps) - len(smoothed):]
        ax.plot(smooth_steps, smoothed, color="red", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy Over Training")
    ax.axhline(y=np.log(500), color="gray", linestyle="--", alpha=0.5, label="Max (log 500)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.plot_dir / "entropy.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_set_comparison(config: Config, baseline_results: Optional[Dict] = None):
    """Plot per-reference-set grouped bar chart: agent vs baselines."""
    eval_results_path = config.log_dir / "eval_results.json"
    if not eval_results_path.exists():
        print("No evaluation results found.")
        return

    with open(eval_results_path) as f:
        eval_results = json.load(f)

    per_set = eval_results.get("per_set", {})
    if not per_set:
        return

    ref_sets = sorted(per_set.keys())
    agent_rewards = [per_set[rs]["reward_mean"] for rs in ref_sets]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ref_sets))
    width = 0.35

    ax.bar(x - width / 2, agent_rewards, width, label="RL Agent", color="steelblue")

    if baseline_results:
        baseline_rewards = [
            baseline_results.get(f"{rs}_reward", 0.0) for rs in ref_sets
        ]
        ax.bar(x + width / 2, baseline_rewards, width, label="Default BLOSUM62", color="coral")

    ax.set_xlabel("Reference Set")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Agent vs Baseline by Reference Set")
    ax.set_xticks(x)
    ax.set_xticklabels(ref_sets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = config.plot_dir / "per_set_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_action_heatmap(config: Config):
    """Plot 2D heatmap of action selection frequency (gap_open x gap_extend)."""
    eval_results_path = config.log_dir / "eval_results.json"
    if not eval_results_path.exists():
        return

    with open(eval_results_path) as f:
        eval_results = json.load(f)

    action_dist = eval_results.get("action_distribution", {})
    top_actions = action_dist.get("top_10", [])
    if not top_actions:
        return

    action_space = ActionSpace(config)

    # Build 2D frequency grid
    freq = np.zeros((config.gap_open_bins, config.gap_extend_bins))
    for entry in top_actions:
        action_idx = entry["action"]
        count = entry["count"]
        # Find bin indices
        go, ge, _ = action_space.decode(action_idx)
        i = np.argmin(np.abs(action_space.gap_open_values - go))
        j = np.argmin(np.abs(action_space.gap_extend_values - ge))
        freq[i, j] += count

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(freq, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xlabel("Gap Extend Bin")
    ax.set_ylabel("Gap Open Bin")
    ax.set_title("Action Frequency: Gap Open x Gap Extend")

    # Label axes with actual values
    ax.set_xticks(range(config.gap_extend_bins))
    ax.set_xticklabels([f"{v:.1f}" for v in action_space.gap_extend_values], rotation=45)
    ax.set_yticks(range(config.gap_open_bins))
    ax.set_yticklabels([f"{v:.1f}" for v in action_space.gap_open_values])

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    save_path = config.plot_dir / "action_heatmap.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_matrix_distribution(config: Config):
    """Pie chart of substitution matrix selection."""
    eval_results_path = config.log_dir / "eval_results.json"
    if not eval_results_path.exists():
        return

    with open(eval_results_path) as f:
        eval_results = json.load(f)

    matrix_counts = eval_results.get("action_distribution", {}).get("matrices", {})
    if not matrix_counts:
        return

    labels = list(matrix_counts.keys())
    sizes = list(matrix_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Substitution Matrix Selection")

    plt.tight_layout()
    save_path = config.plot_dir / "matrix_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(config: Config, baseline_results: Optional[Dict] = None):
    """Generate all visualization plots."""
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")
    plot_learning_curves(config)
    plot_entropy(config)
    plot_per_set_comparison(config, baseline_results)
    plot_action_heatmap(config)
    plot_matrix_distribution(config)
    print("All plots generated.")
