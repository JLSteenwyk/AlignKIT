"""Generate experimental workflow diagram for the RL-MSA system.

Run:
    python results/msa_continuous/workflow_diagram.py

Produces workflow_diagram.pdf and workflow_diagram.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colors
COL_DATA = "#E8F4FD"       # light blue
COL_FEATURES = "#FFF3E0"   # light orange
COL_POLICY = "#E8F5E9"     # light green
COL_MAFFT = "#F3E5F5"      # light purple
COL_SCORE = "#FCE4EC"      # light red
COL_UPDATE = "#FFFDE7"     # light yellow
COL_EVAL = "#E0F2F1"       # light teal
BORDER = "#333333"
ARROW_COL = "#555555"


def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold_title=None, border_color=BORDER):
    """Draw a rounded box with optional bold title line."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=border_color, linewidth=1.2,
    )
    ax.add_patch(box)

    if bold_title:
        ax.text(x, y + h * 0.15, bold_title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#222222")
        ax.text(x, y - h * 0.15, text, ha="center", va="center",
                fontsize=fontsize - 1, color="#444444")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#222222")


def draw_arrow(ax, x1, y1, x2, y2, label=None, color=ARROW_COL, style="->"):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color,
            lw=1.5, connectionstyle="arc3,rad=0",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label, fontsize=6.5, color="#666666",
                ha="left", va="center", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5.0, 7.2, "Experimental Workflow: RL-Learned Gap Penalties for MSA",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#111111")

    # =========================================================================
    # Top section: TRAINING LOOP
    # =========================================================================
    ax.text(5.0, 6.6, "Training Loop (Contextual Bandit)", ha="center",
            va="center", fontsize=10, fontweight="bold", color="#4C72B0",
            style="italic")

    # Step 1: Dataset
    draw_box(ax, 1.2, 5.5, 2.0, 1.0, "BAliBASE 3.0\nRV11/RV12/RV20/RV30\n153 MSA cases",
             COL_DATA, fontsize=8, bold_title="1. Sample Case")

    # Step 2: Feature extraction
    draw_box(ax, 3.8, 5.5, 1.8, 1.0,
             "20-dim vector:\nlengths, composition,\ncomplexity, similarity",
             COL_FEATURES, fontsize=7.5, bold_title="2. Extract Features")

    # Step 3: Policy network
    draw_box(ax, 6.4, 5.5, 1.8, 1.0,
             "Gaussian MLP\n20 -> 128 -> 128 -> 2\nop, ep prediction",
             COL_POLICY, fontsize=7.5, bold_title="3. Policy Network")

    # Step 4: MAFFT
    draw_box(ax, 9.0, 5.5, 1.8, 1.0,
             "MAFFT L-INS-i\n--maxiterate 10\n--op {op} --ep {ep}",
             COL_MAFFT, fontsize=7.5, bold_title="4. Run MAFFT")

    # Step 5: Score
    draw_box(ax, 9.0, 3.5, 1.8, 1.0,
             "SP + TC scores\nvs BAliBASE reference\nR = 0.5*SP + 0.5*TC",
             COL_SCORE, fontsize=7.5, bold_title="5. Score Alignment")

    # Step 6: Update
    draw_box(ax, 5.1, 3.5, 2.4, 1.0,
             "REINFORCE + value baseline\nadvantage normalization\nentropy bonus, grad clip",
             COL_UPDATE, fontsize=7.5, bold_title="6. Policy Update")

    # Arrows for training loop
    draw_arrow(ax, 2.2, 5.5, 2.9, 5.5, "sequences")
    draw_arrow(ax, 4.7, 5.5, 5.5, 5.5, "state (20-d)")
    draw_arrow(ax, 7.3, 5.5, 8.1, 5.5, "op, ep")
    draw_arrow(ax, 9.0, 5.0, 9.0, 4.0, "predicted\nMSA")
    draw_arrow(ax, 8.1, 3.5, 6.3, 3.5, "reward")

    # Loop-back arrow from Update to Policy
    ax.annotate(
        "", xy=(5.5, 5.0), xytext=(5.1, 4.0),
        arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.5,
                        connectionstyle="arc3,rad=-0.3"),
    )
    ax.text(4.4, 4.5, "update\nweights", fontsize=6.5, color="#4C72B0",
            ha="center", va="center", style="italic")

    # Loop-back arrow from Dataset to itself (sampling)
    ax.annotate(
        "", xy=(0.5, 5.0), xytext=(3.9, 3.0),
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0,
                        connectionstyle="arc3,rad=0.4", linestyle="dashed"),
    )
    ax.text(1.3, 3.5, "repeat\n10K episodes", fontsize=6.5, color="#999999",
            ha="center", va="center", style="italic")

    # =========================================================================
    # Bottom section: EVALUATION
    # =========================================================================
    ax.text(5.0, 2.0, "Evaluation", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#DD8452", style="italic")

    # Eval dataset
    draw_box(ax, 1.2, 1.0, 2.0, 0.9,
             "BAliBASE 3.0\nRV40 (49) + RV50 (16)\n65 held-out cases",
             COL_EVAL, fontsize=7.5, bold_title="Eval Dataset")

    # Eval methods
    draw_box(ax, 4.5, 1.0, 2.6, 0.9,
             "MAFFT --auto | L-INS-i default | RL Agent (greedy)",
             COL_EVAL, fontsize=7.5, bold_title="Compare Methods")

    # Eval metrics
    draw_box(ax, 8.2, 1.0, 2.6, 0.9,
             "Reward: 0.446 | 0.448 | 0.454\n"
             "TC:     0.193 | 0.194 | 0.203",
             COL_EVAL, fontsize=7.5, bold_title="Results (+2.0%)")

    draw_arrow(ax, 2.2, 1.0, 3.2, 1.0)
    draw_arrow(ax, 5.8, 1.0, 6.9, 1.0)

    # =========================================================================
    # Side annotations
    # =========================================================================

    # Action space box
    draw_box(ax, 1.2, 3.5, 2.0, 0.8,
             "op in [0.5, 5.0]\nep in [0.0, 1.0]",
             "#F5F5F5", fontsize=7.5, bold_title="Action Space",
             border_color="#AAAAAA")

    # Key insight box
    box_insight = FancyBboxPatch(
        (0.0, -0.3), 10.0, 0.5,
        boxstyle="round,pad=0.1",
        facecolor="#FFF8E1", edgecolor="#FFA000", linewidth=1.0,
    )
    ax.add_patch(box_insight)
    ax.text(5.0, -0.05,
            "Key insight: The agent learns op ~ 3.0 (2x default) and ep ~ 0.4 (vs default 0.0), "
            "yielding +2.0% reward and +5.4% TC improvement",
            ha="center", va="center", fontsize=8, color="#333333", style="italic")

    # Save
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, "workflow_diagram.%s" % ext)
        fig.savefig(path)
        print("  saved %s" % path)
    plt.close(fig)


if __name__ == "__main__":
    main()
