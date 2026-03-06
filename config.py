"""Configuration for pairwise and MSA RL gap penalty learning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Hyperparameters and paths for pairwise RL training."""

    # Regions
    num_regions: int = 3  # K: N-terminal, middle, C-terminal
    action_dim: int = 6   # 2 * num_regions (gap_open + gap_extend per region)
    input_dim: int = 12   # pairwise feature dimension

    # Action bounds
    gap_open_min: float = 1.0
    gap_open_max: float = 20.0
    gap_extend_min: float = 0.1
    gap_extend_max: float = 5.0

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])

    # Optimization
    lr: float = 3e-4
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    grad_clip_max_norm: float = 1.0

    # Policy
    initial_log_std: float = 0.0   # std=1.0 initially
    min_std: float = 0.05

    # Training
    num_episodes: int = 50000
    batch_size: int = 32
    seed: int = 42

    # Scoring
    sp_weight: float = 0.5
    tc_weight: float = 0.5

    # Data
    data_dir: Path = Path("data/bb3_release")
    max_seq_length: int = 500
    train_ref_sets: List[str] = field(default_factory=lambda: ["RV11", "RV12", "RV20", "RV30"])
    eval_ref_sets: List[str] = field(default_factory=lambda: ["RV40", "RV50"])

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints_pw")
    checkpoint_every: int = 500
    eval_every: int = 500

    # Logging
    log_dir: Path = Path("logs_pw")


@dataclass
class MSAConfig:
    """Hyperparameters and paths for MSA RL training."""

    # Policy
    input_dim: int = 20        # MSA feature dimension
    action_dim: int = 2        # (op, ep) for MAFFT
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    initial_log_std: float = 0.0
    min_std: float = 0.05

    # Action bounds (MAFFT op and ep)
    op_min: float = 0.5
    op_max: float = 5.0
    ep_min: float = 0.0
    ep_max: float = 1.0

    # Optimization
    lr: float = 3e-4
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    grad_clip_max_norm: float = 1.0

    # Training
    num_episodes: int = 10000
    batch_size: int = 16
    seed: int = 42

    # Scoring
    msa_sp_weight: float = 0.5
    msa_tc_weight: float = 0.5

    # Data
    data_dir: Path = Path("data/bb3_release")
    train_ref_sets: List[str] = field(default_factory=lambda: ["RV11", "RV12", "RV20", "RV30"])
    eval_ref_sets: List[str] = field(default_factory=lambda: ["RV40", "RV50"])

    # MAFFT
    mafft_bin: Path = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mafft-7.525-with-extensions/bin/mafft")
    mafft_threads: int = 1
    mafft_timeout: int = 120

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints_msa")
    checkpoint_every: int = 100
    eval_every: int = 200

    # Logging
    log_dir: Path = Path("logs_msa")
