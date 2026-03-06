"""Configuration dataclass for MSA-level RL alignment with MAFFT and MUSCLE."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MSAConfig:
    # --- Paths ---
    project_dir: Path = Path(__file__).parent
    data_dir: Path = field(default=None)
    benchmark_dir: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)
    log_dir: Path = field(default=None)
    plot_dir: Path = field(default=None)

    # Tool binaries
    mafft_bin: Path = Path(
        "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mafft-7.525-with-extensions/bin/mafft"
    )
    muscle_bin: Path = Path(
        "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/muscle_v5.3/muscle-linux-x86.v5.3"
    )
    clustalo_bin: Path = Path(
        "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/clustalo/clustalo-1.2.4-Ubuntu-x86_64"
    )

    # --- MAFFT action space ---
    mafft_op_values: List[float] = field(
        default_factory=lambda: [1.0, 1.53, 2.5, 4.0]
    )
    mafft_ep_values: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.5]
    )
    mafft_strategies: List[str] = field(
        default_factory=lambda: ["auto", "localpair", "globalpair", "genafpair"]
    )
    mafft_maxiterate_values: List[int] = field(
        default_factory=lambda: [0, 2, 10]
    )

    # --- MUSCLE action space ---
    muscle_commands: List[str] = field(
        default_factory=lambda: ["align"]
    )
    muscle_perm_values: List[str] = field(
        default_factory=lambda: ["none", "abc", "bca"]
    )
    muscle_perturb_values: List[int] = field(
        default_factory=lambda: [0, 1, 3]
    )

    # --- Clustal Omega action space ---
    clustalo_iter_values: List[int] = field(
        default_factory=lambda: [0, 1, 3, 5]
    )
    clustalo_full_values: List[bool] = field(
        default_factory=lambda: [False, True]
    )
    clustalo_full_iter_values: List[bool] = field(
        default_factory=lambda: [False, True]
    )
    clustalo_kimura_values: List[bool] = field(
        default_factory=lambda: [False, True]
    )

    # --- RL hyperparameters ---
    lr: float = 1e-3
    gamma: float = 1.0  # bandit — no discounting
    entropy_coeff: float = 0.01
    tool_entropy_coeff: float = 0.01   # reduced to allow tool specialization
    param_entropy_coeff: float = 0.01
    baseline_momentum: float = 0.99
    grad_clip_max_norm: float = 1.0
    value_coeff: float = 0.5  # weight for value head MSE loss

    # --- Policy architecture ---
    policy_type: str = "flat"  # "hierarchical" or "flat"
    mafft_only: bool = True  # restrict action space to MAFFT params only

    # --- Training ---
    num_episodes: int = 10000
    batch_size: int = 8
    eval_every: int = 500
    log_every: int = 20
    checkpoint_every: int = 500
    seed: int = 42

    # --- Subprocess ---
    subprocess_timeout: Optional[int] = 600  # seconds (10 minutes)
    mafft_threads: int = 4

    # --- Network ---
    input_dim: int = 20
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])

    # --- Reward ---
    sp_weight: float = 0.5
    tc_weight: float = 0.5

    # --- Dataset ---
    all_ref_sets: List[str] = field(
        default_factory=lambda: ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]
    )
    use_benchmarks: bool = False  # BAliBASE only
    split_ratio: float = 0.8  # fraction used for training
    split_seed: int = 42      # reproducible stratified split

    # Subsample each external benchmark (HOMSTRAD, OXBench, SABRE) to this
    # many training cases using farthest-point sampling in feature space.
    # Set to None to use all available cases (original behavior).
    # Set to e.g. 172 to match the total BAliBASE training count per database.
    subsample_benchmarks_to: Optional[int] = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_dir / "data" / "bb3_release"
        if self.benchmark_dir is None:
            self.benchmark_dir = self.project_dir / "data" / "benchmarks"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_dir / "checkpoints_msa"
        if self.log_dir is None:
            self.log_dir = self.project_dir / "logs_msa"
        if self.plot_dir is None:
            self.plot_dir = self.project_dir / "plots_msa"

    @property
    def n_mafft_actions(self) -> int:
        return (
            len(self.mafft_op_values)
            * len(self.mafft_ep_values)
            * len(self.mafft_strategies)
            * len(self.mafft_maxiterate_values)
        )

    @property
    def n_muscle_actions(self) -> int:
        return (
            len(self.muscle_commands)
            * len(self.muscle_perm_values)
            * len(self.muscle_perturb_values)
        )

    @property
    def n_clustalo_actions(self) -> int:
        return (
            len(self.clustalo_iter_values)
            * len(self.clustalo_full_values)
            * len(self.clustalo_full_iter_values)
            * len(self.clustalo_kimura_values)
        )

    @property
    def num_actions(self) -> int:
        if self.mafft_only:
            return self.n_mafft_actions
        return self.n_mafft_actions + self.n_muscle_actions + self.n_clustalo_actions

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)
