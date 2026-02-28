"""Central configuration dataclass for the RL alignment framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # --- Paths ---
    project_dir: Path = Path(__file__).parent
    data_dir: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)
    log_dir: Path = field(default=None)
    plot_dir: Path = field(default=None)

    # --- Action space ---
    # Gap open: 10 bins linearly spaced in [-20, -1]
    gap_open_min: float = -20.0
    gap_open_max: float = -1.0
    gap_open_bins: int = 10
    # Gap extend: 10 bins linearly spaced in [-5, -0.1]
    gap_extend_min: float = -5.0
    gap_extend_max: float = -0.1
    gap_extend_bins: int = 10
    # Substitution matrices
    matrices: List[str] = field(
        default_factory=lambda: ["BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM70", "PAM250"]
    )

    # --- RL hyperparameters ---
    lr: float = 1e-3
    gamma: float = 1.0  # bandit — no discounting
    entropy_coeff: float = 0.01
    baseline_momentum: float = 0.99
    grad_clip_max_norm: float = 1.0

    # --- Training ---
    num_episodes: int = 10000
    batch_size: int = 32
    max_seq_length: int = 2000
    eval_every: int = 500
    log_every: int = 100
    checkpoint_every: int = 1000
    seed: int = 42

    # --- Network ---
    input_dim: int = 5
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])

    # --- Reward ---
    sp_weight: float = 0.5
    tc_weight: float = 0.5

    # --- Dataset ---
    train_ref_sets: List[str] = field(
        default_factory=lambda: ["RV11", "RV12", "RV20", "RV30"]
    )
    eval_ref_sets: List[str] = field(
        default_factory=lambda: ["RV40", "RV50"]
    )

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / "bb3_release"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_dir / "checkpoints"
        if self.log_dir is None:
            self.log_dir = self.project_dir / "logs"
        if self.plot_dir is None:
            self.plot_dir = self.project_dir / "plots"

    @property
    def num_actions(self) -> int:
        return self.gap_open_bins * self.gap_extend_bins * len(self.matrices)

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)
