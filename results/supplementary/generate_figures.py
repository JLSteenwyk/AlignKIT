"""Generate supplementary figures for the oracle analysis.

Run:
    python results/supplementary/generate_figures.py

Produces Fig S1–S6 as PDF and PNG in this directory.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

from config_msa import MSAConfig
from env.msa_action_space import MSAActionSpace

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, "..", "oracle_analysis", "oracle_results.npz")

COLORS = {"mafft": "#4C72B0", "muscle": "#DD8452", "clustalo": "#55A868"}
TOOL_LABELS = {"mafft": "MAFFT", "muscle": "MUSCLE", "clustalo": "Clustal Ω"}

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
    "pdf.fonttype": 42,  # TrueType for editability
    "ps.fonttype": 42,
})


def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  saved {path}")
    plt.close(fig)


def load_data():
    """Load oracle results and build action space."""
    data = np.load(DATA_PATH, allow_pickle=True)
    config = MSAConfig()
    action_space = MSAActionSpace(config)
    return data, config, action_space


# ---------------------------------------------------------------------------
# Fig S1: Tool Performance Comparison
# ---------------------------------------------------------------------------

def fig_s1(data, action_space):
    """Per-tool oracle reward vs per-tool best-fixed reward."""
    rewards = data["rewards"]
    n_mafft = action_space.n_mafft
    n_muscle = action_space.n_muscle
    mafft_end = n_mafft
    muscle_end = n_mafft + n_muscle

    slices = {
        "mafft": (0, mafft_end),
        "muscle": (mafft_end, muscle_end),
        "clustalo": (muscle_end, rewards.shape[1]),
    }

    tools = ["mafft", "muscle", "clustalo"]
    oracle_means, oracle_ses = [], []
    fixed_means = []

    for t in tools:
        lo, hi = slices[t]
        r = rewards[:, lo:hi]
        per_case_best = np.nanmax(r, axis=1)
        oracle_means.append(np.nanmean(per_case_best))
        oracle_ses.append(np.nanstd(per_case_best) / np.sqrt(len(per_case_best)))
        fixed_means.append(np.nanmax(np.nanmean(r, axis=0)))

    x = np.arange(len(tools))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars1 = ax.bar(x - width / 2, oracle_means, width, yerr=oracle_ses,
                   capsize=3, label="Per-case oracle",
                   color=[COLORS[t] for t in tools], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, fixed_means, width,
                   label="Best fixed config",
                   color=[COLORS[t] for t in tools], edgecolor="white", linewidth=0.5,
                   alpha=0.45)

    ax.set_ylabel("Mean reward")
    ax.set_xticks(x)
    ax.set_xticklabels([TOOL_LABELS[t] for t in tools])
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax.set_ylim(0.35, 0.60)
    ax.set_title("Fig S1: Tool Performance Comparison")

    save_fig(fig, "fig_s1_tool_comparison")


# ---------------------------------------------------------------------------
# Fig S2: Headroom Decomposition (waterfall)
# ---------------------------------------------------------------------------

def fig_s2(data, action_space):
    """Waterfall: best-fixed → MAFFT oracle → full oracle."""
    rewards = data["rewards"]
    n_mafft = action_space.n_mafft

    best_fixed = np.nanmax(np.nanmean(rewards, axis=0))
    mafft_oracle = np.nanmean(np.nanmax(rewards[:, :n_mafft], axis=1))
    full_oracle = np.nanmean(np.nanmax(rewards, axis=1))

    total_headroom = full_oracle - best_fixed
    mafft_gain = mafft_oracle - best_fixed
    cross_tool_gain = full_oracle - mafft_oracle
    mafft_pct = 100 * mafft_gain / total_headroom if total_headroom > 0 else 0
    cross_pct = 100 * cross_tool_gain / total_headroom if total_headroom > 0 else 0

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    labels = ["Best fixed\nconfig", "MAFFT param\ntuning", "Cross-tool\nselection", "Full\noracle"]
    values = [best_fixed, mafft_gain, cross_tool_gain, full_oracle]
    bottoms = [0, best_fixed, mafft_oracle, 0]
    colors = ["#999999", COLORS["mafft"], "#8B5CF6", "#333333"]

    for i, (lbl, val, bot, col) in enumerate(zip(labels, values, bottoms, colors)):
        bar = ax.bar(i, val, bottom=bot, color=col, width=0.6, edgecolor="white", linewidth=0.5)
        if i == 0 or i == 3:
            ax.text(i, val + bot + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        else:
            ax.text(i, val + bot + 0.002, f"+{val:.4f}\n({mafft_pct:.0f}%)" if i == 1 else f"+{val:.4f}\n({cross_pct:.0f}%)",
                    ha="center", va="bottom", fontsize=7)

    # Connector lines
    for i in range(3):
        top = bottoms[i] + values[i] if i < 3 else values[i]
        if i == 0:
            top = values[0]
        elif i == 1:
            top = best_fixed + mafft_gain
        elif i == 2:
            top = mafft_oracle + cross_tool_gain
        ax.plot([i + 0.3, i + 0.7], [top, top], color="0.5", linewidth=0.8, linestyle="--")

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean reward")
    ax.set_ylim(0.42, 0.56)
    ax.set_title("Fig S2: Headroom Decomposition")

    save_fig(fig, "fig_s2_headroom_waterfall")


# ---------------------------------------------------------------------------
# Fig S3: Per-Dataset Oracle Breakdown
# ---------------------------------------------------------------------------

def fig_s3(data, action_space):
    """Per-ref_set oracle vs best-fixed, with tool distribution coloring."""
    rewards = data["rewards"]
    case_ref_sets = list(data["case_ref_sets"])
    n_mafft = action_space.n_mafft
    n_muscle = action_space.n_muscle
    mafft_end = n_mafft
    muscle_end = n_mafft + n_muscle

    all_rs = sorted(set(case_ref_sets))

    oracle_vals = []
    fixed_vals = []
    tool_fracs = []  # fraction mafft/muscle/clustalo per ref_set

    for rs in all_rs:
        mask = [j for j, r in enumerate(case_ref_sets) if r == rs]
        rs_rewards = rewards[mask]
        oracle_vals.append(np.nanmean(np.nanmax(rs_rewards, axis=1)))
        fixed_vals.append(np.nanmax(np.nanmean(rs_rewards, axis=0)))

        best_per_case = np.nanargmax(rs_rewards, axis=1)
        n = len(mask)
        n_m = sum(1 for a in best_per_case if a < mafft_end)
        n_u = sum(1 for a in best_per_case if mafft_end <= a < muscle_end)
        n_c = sum(1 for a in best_per_case if a >= muscle_end)
        tool_fracs.append((n_m / n, n_u / n, n_c / n))

    x = np.arange(len(all_rs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    ax.bar(x - width / 2, fixed_vals, width, label="Best fixed config",
           color="#999999", edgecolor="white", linewidth=0.5)

    # Oracle bars stacked by tool fraction
    for i, rs in enumerate(all_rs):
        fm, fu, fc = tool_fracs[i]
        h = oracle_vals[i]
        bot = 0
        for frac, col, tool in [(fm, COLORS["mafft"], "mafft"),
                                  (fu, COLORS["muscle"], "muscle"),
                                  (fc, COLORS["clustalo"], "clustalo")]:
            seg = h * frac
            ax.bar(i + width / 2, seg, width, bottom=bot, color=col,
                   edgecolor="white", linewidth=0.5)
            bot += seg

    # Manual legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#999999", label="Best fixed"),
        Patch(facecolor=COLORS["mafft"], label="Oracle (MAFFT)"),
        Patch(facecolor=COLORS["muscle"], label="Oracle (MUSCLE)"),
        Patch(facecolor=COLORS["clustalo"], label="Oracle (Clustal Ω)"),
    ]
    ax.legend(handles=legend_elems, frameon=True, fancybox=False, edgecolor="0.8",
              fontsize=7, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(all_rs)
    ax.set_ylabel("Mean reward")
    ax.set_xlabel("BAliBASE reference set")
    ax.set_title("Fig S3: Per-Dataset Oracle Breakdown")

    save_fig(fig, "fig_s3_per_dataset")


# ---------------------------------------------------------------------------
# Fig S4: MAFFT Parameter Sensitivity (4 panels)
# ---------------------------------------------------------------------------

def fig_s4(data, config, action_space):
    """4-panel figure showing marginal effect of each MAFFT parameter."""
    rewards = data["rewards"]
    n_mafft = action_space.n_mafft
    mafft_rewards = rewards[:, :n_mafft]  # (42, 144)

    # Decode all MAFFT actions to get their parameters
    ops, eps, strats, maxiters = [], [], [], []
    for a in range(n_mafft):
        act = action_space.decode(a)
        ops.append(act.mafft_op)
        eps.append(act.mafft_ep)
        strats.append(act.mafft_strategy)
        maxiters.append(act.mafft_maxiterate)

    ops = np.array(ops)
    eps = np.array(eps)
    strats = np.array(strats)
    maxiters = np.array(maxiters)

    # Mean reward per action across all cases
    mean_per_action = np.nanmean(mafft_rewards, axis=0)  # (144,)

    param_info = [
        ("Opening penalty (op)", ops, config.mafft_op_values),
        ("Extension penalty (ep)", eps, config.mafft_ep_values),
        ("Strategy", strats, config.mafft_strategies),
        ("Max iterations", maxiters, config.mafft_maxiterate_values),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)

    for ax, (title, param_arr, param_vals) in zip(axes, param_info):
        positions = []
        box_data = []
        labels = []
        for pv in param_vals:
            if isinstance(pv, str):
                mask = param_arr == pv
            else:
                mask = np.isclose(param_arr, pv) if isinstance(pv, float) else (param_arr == pv)
            box_data.append(mean_per_action[mask])
            labels.append(str(pv))
            positions.append(len(positions))

        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(facecolor=COLORS["mafft"], alpha=0.6),
                        flierprops=dict(markersize=3))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30 if len(labels[0]) > 3 else 0, fontsize=7)
        ax.set_title(title, fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Mean reward\n(across cases)")

    fig.suptitle("Fig S4: MAFFT Parameter Sensitivity", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_s4_mafft_sensitivity")


# ---------------------------------------------------------------------------
# Fig S5: Oracle Action Diversity
# ---------------------------------------------------------------------------

def fig_s5(data, action_space):
    """Heatmap: 42 cases × their oracle-best action, colored by tool."""
    rewards = data["rewards"]
    case_ids = list(data["case_ids"])
    case_ref_sets = list(data["case_ref_sets"])
    n_mafft = action_space.n_mafft
    n_muscle = action_space.n_muscle
    mafft_end = n_mafft
    muscle_end = n_mafft + n_muscle

    best_actions = np.nanargmax(rewards, axis=1)
    unique_actions = sorted(set(best_actions))
    n_unique = len(unique_actions)
    action_to_col = {a: i for i, a in enumerate(unique_actions)}

    # Build binary matrix (42 × n_unique)
    matrix = np.zeros((len(case_ids), n_unique))
    for i, a in enumerate(best_actions):
        matrix[i, action_to_col[a]] = 1.0

    # Color by tool
    from matplotlib.colors import ListedColormap
    tool_colors = []
    for a in unique_actions:
        if a < mafft_end:
            tool_colors.append(COLORS["mafft"])
        elif a < muscle_end:
            tool_colors.append(COLORS["muscle"])
        else:
            tool_colors.append(COLORS["clustalo"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for i in range(len(case_ids)):
        a = best_actions[i]
        col_idx = action_to_col[a]
        ax.scatter(col_idx, i, color=tool_colors[col_idx], s=30, marker="s", edgecolors="white", linewidth=0.3)

    ax.set_yticks(range(len(case_ids)))
    ylabels = [f"{case_ref_sets[i]}:{case_ids[i]}" for i in range(len(case_ids))]
    ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.set_xlabel(f"Oracle-best action (sorted, {n_unique} unique)")
    ax.set_ylabel("Evaluation case")
    ax.invert_yaxis()

    # Column labels (action indices)
    ax.set_xticks(range(n_unique))
    ax.set_xticklabels([str(a) for a in unique_actions], fontsize=5, rotation=90)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLORS["mafft"], label="MAFFT"),
        Patch(facecolor=COLORS["muscle"], label="MUSCLE"),
        Patch(facecolor=COLORS["clustalo"], label="Clustal Ω"),
    ]
    ax.legend(handles=legend_elems, frameon=True, fancybox=False, edgecolor="0.8",
              fontsize=7, loc="lower right")

    ax.set_title(f"Fig S5: Oracle Action Diversity ({n_unique} unique best actions across 42 cases)")
    fig.tight_layout()
    save_fig(fig, "fig_s5_action_diversity")


# ---------------------------------------------------------------------------
# Fig S6: Reward Distribution by Dataset
# ---------------------------------------------------------------------------

def fig_s6(data):
    """Violin/box plot of per-case oracle reward grouped by ref_set."""
    rewards = data["rewards"]
    case_ref_sets = list(data["case_ref_sets"])
    all_rs = sorted(set(case_ref_sets))

    per_case_oracle = np.nanmax(rewards, axis=1)

    groups = []
    for rs in all_rs:
        mask = [j for j, r in enumerate(case_ref_sets) if r == rs]
        groups.append(per_case_oracle[mask])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    parts = ax.violinplot(groups, positions=range(len(all_rs)), showmedians=False,
                          showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["mafft"])
        body.set_alpha(0.3)

    bp = ax.boxplot(groups, positions=range(len(all_rs)), widths=0.3, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(facecolor=COLORS["mafft"], alpha=0.5),
                    flierprops=dict(markersize=3),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    ax.set_xticks(range(len(all_rs)))
    ax.set_xticklabels(all_rs)
    ax.set_ylabel("Oracle reward (per case)")
    ax.set_xlabel("BAliBASE reference set")
    ax.set_title("Fig S6: Reward Distribution by Dataset")
    fig.tight_layout()

    save_fig(fig, "fig_s6_reward_distributions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    data, config, action_space = load_data()
    print(f"  {data['rewards'].shape[0]} cases × {data['rewards'].shape[1]} actions")

    print("\nGenerating Fig S1: Tool Performance Comparison")
    fig_s1(data, action_space)

    print("\nGenerating Fig S2: Headroom Decomposition")
    fig_s2(data, action_space)

    print("\nGenerating Fig S3: Per-Dataset Oracle Breakdown")
    fig_s3(data, action_space)

    print("\nGenerating Fig S4: MAFFT Parameter Sensitivity")
    fig_s4(data, config, action_space)

    print("\nGenerating Fig S5: Oracle Action Diversity")
    fig_s5(data, action_space)

    print("\nGenerating Fig S6: Reward Distribution by Dataset")
    fig_s6(data)

    print("\nDone — all figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
