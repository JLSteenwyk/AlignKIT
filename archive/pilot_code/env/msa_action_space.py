"""Discrete action space for MSA tool+parameter selection.

Maps a flat action index to a MAFFT, MUSCLE, or Clustal Omega configuration.

Index layout:
    0 .. n_mafft-1                          = MAFFT actions
    n_mafft .. n_mafft+n_muscle-1           = MUSCLE actions
    n_mafft+n_muscle .. n_total-1           = Clustal Omega actions

MAFFT: op_idx * (n_ep * n_strat * n_maxiter) + ep_idx * (n_strat * n_maxiter) + strat_idx * n_maxiter + maxiter_idx
MUSCLE: cmd_idx * (n_perm * n_perturb) + perm_idx * n_perturb + perturb_idx
Clustal Omega: iter_idx * (n_full * n_full_iter * n_kimura) + full_idx * (n_full_iter * n_kimura) + full_iter_idx * n_kimura + kimura_idx
"""

from dataclasses import dataclass
from typing import List, Optional

from config_msa import MSAConfig


@dataclass
class MSAAction:
    """Decoded MSA action with tool name and tool-specific parameters."""
    tool: str  # "mafft", "muscle", or "clustalo"

    # MAFFT params (None unless tool == "mafft")
    mafft_op: Optional[float] = None
    mafft_ep: Optional[float] = None
    mafft_strategy: Optional[str] = None
    mafft_maxiterate: Optional[int] = None

    # MUSCLE params (None unless tool == "muscle")
    muscle_command: Optional[str] = None
    muscle_perm: Optional[str] = None
    muscle_perturb: Optional[int] = None

    # Clustal Omega params (None unless tool == "clustalo")
    clustalo_iter: Optional[int] = None
    clustalo_full: Optional[bool] = None
    clustalo_full_iter: Optional[bool] = None
    clustalo_kimura: Optional[bool] = None


# Tool index constants
TOOL_MAFFT = 0
TOOL_MUSCLE = 1
TOOL_CLUSTALO = 2
TOOL_NAMES = ["mafft", "muscle", "clustalo"]


class MSAActionSpace:
    """Maps flat action indices to MSA tool configurations.

    Also supports hierarchical (tool_idx, param_idx) decoding.
    """

    def __init__(self, config: MSAConfig):
        self.config = config

        # MAFFT dimensions
        self.mafft_ops = list(config.mafft_op_values)
        self.mafft_eps = list(config.mafft_ep_values)
        self.mafft_strats = list(config.mafft_strategies)
        self.mafft_maxiters = list(config.mafft_maxiterate_values)

        self.n_mafft_op = len(self.mafft_ops)
        self.n_mafft_ep = len(self.mafft_eps)
        self.n_mafft_strat = len(self.mafft_strats)
        self.n_mafft_maxiter = len(self.mafft_maxiters)
        self.n_mafft = self.n_mafft_op * self.n_mafft_ep * self.n_mafft_strat * self.n_mafft_maxiter

        # MUSCLE dimensions
        self.muscle_cmds = list(config.muscle_commands)
        self.muscle_perms = list(config.muscle_perm_values)
        self.muscle_perturbs = list(config.muscle_perturb_values)

        self.n_muscle_cmd = len(self.muscle_cmds)
        self.n_muscle_perm = len(self.muscle_perms)
        self.n_muscle_perturb = len(self.muscle_perturbs)
        self.n_muscle = self.n_muscle_cmd * self.n_muscle_perm * self.n_muscle_perturb

        # Clustal Omega dimensions
        self.clustalo_iters = list(config.clustalo_iter_values)
        self.clustalo_fulls = list(config.clustalo_full_values)
        self.clustalo_full_iters = list(config.clustalo_full_iter_values)
        self.clustalo_kimuras = list(config.clustalo_kimura_values)

        self.n_clustalo_iter = len(self.clustalo_iters)
        self.n_clustalo_full = len(self.clustalo_fulls)
        self.n_clustalo_full_iter = len(self.clustalo_full_iters)
        self.n_clustalo_kimura = len(self.clustalo_kimuras)
        self.n_clustalo = (self.n_clustalo_iter * self.n_clustalo_full
                           * self.n_clustalo_full_iter * self.n_clustalo_kimura)

        self.n_actions = self.n_mafft + self.n_muscle + self.n_clustalo

    @property
    def param_sizes(self) -> List[int]:
        """Number of parameter combinations per tool: [n_mafft, n_muscle, n_clustalo]."""
        return [self.n_mafft, self.n_muscle, self.n_clustalo]

    # --- Private decode helpers ---

    def _decode_mafft_param(self, param_idx: int) -> MSAAction:
        """Decode a MAFFT-local param index to MSAAction."""
        assert 0 <= param_idx < self.n_mafft, (
            f"MAFFT param_idx {param_idx} out of range [0, {self.n_mafft})"
        )
        idx = param_idx
        maxiter_idx = idx % self.n_mafft_maxiter
        idx //= self.n_mafft_maxiter
        strat_idx = idx % self.n_mafft_strat
        idx //= self.n_mafft_strat
        ep_idx = idx % self.n_mafft_ep
        idx //= self.n_mafft_ep
        op_idx = idx

        return MSAAction(
            tool="mafft",
            mafft_op=self.mafft_ops[op_idx],
            mafft_ep=self.mafft_eps[ep_idx],
            mafft_strategy=self.mafft_strats[strat_idx],
            mafft_maxiterate=self.mafft_maxiters[maxiter_idx],
        )

    def _decode_muscle_param(self, param_idx: int) -> MSAAction:
        """Decode a MUSCLE-local param index to MSAAction."""
        assert 0 <= param_idx < self.n_muscle, (
            f"MUSCLE param_idx {param_idx} out of range [0, {self.n_muscle})"
        )
        idx = param_idx
        perturb_idx = idx % self.n_muscle_perturb
        idx //= self.n_muscle_perturb
        perm_idx = idx % self.n_muscle_perm
        idx //= self.n_muscle_perm
        cmd_idx = idx

        return MSAAction(
            tool="muscle",
            muscle_command=self.muscle_cmds[cmd_idx],
            muscle_perm=self.muscle_perms[perm_idx],
            muscle_perturb=self.muscle_perturbs[perturb_idx],
        )

    def _decode_clustalo_param(self, param_idx: int) -> MSAAction:
        """Decode a ClustalO-local param index to MSAAction."""
        assert 0 <= param_idx < self.n_clustalo, (
            f"ClustalO param_idx {param_idx} out of range [0, {self.n_clustalo})"
        )
        idx = param_idx
        kimura_idx = idx % self.n_clustalo_kimura
        idx //= self.n_clustalo_kimura
        full_iter_idx = idx % self.n_clustalo_full_iter
        idx //= self.n_clustalo_full_iter
        full_idx = idx % self.n_clustalo_full
        idx //= self.n_clustalo_full
        iter_idx = idx

        return MSAAction(
            tool="clustalo",
            clustalo_iter=self.clustalo_iters[iter_idx],
            clustalo_full=self.clustalo_fulls[full_idx],
            clustalo_full_iter=self.clustalo_full_iters[full_iter_idx],
            clustalo_kimura=self.clustalo_kimuras[kimura_idx],
        )

    # --- Public API ---

    def decode(self, action_idx: int) -> MSAAction:
        """Decode flat action index to MSAAction."""
        assert 0 <= action_idx < self.n_actions, (
            f"Action {action_idx} out of range [0, {self.n_actions})"
        )

        if action_idx < self.n_mafft:
            return self._decode_mafft_param(action_idx)
        elif action_idx < self.n_mafft + self.n_muscle:
            return self._decode_muscle_param(action_idx - self.n_mafft)
        else:
            return self._decode_clustalo_param(action_idx - self.n_mafft - self.n_muscle)

    def decode_hierarchical(self, tool_idx: int, param_idx: int) -> MSAAction:
        """Decode a (tool_idx, param_idx) pair to MSAAction.

        Args:
            tool_idx: 0=MAFFT, 1=MUSCLE, 2=ClustalO.
            param_idx: Index within that tool's parameter space.
        """
        if tool_idx == TOOL_MAFFT:
            return self._decode_mafft_param(param_idx)
        elif tool_idx == TOOL_MUSCLE:
            return self._decode_muscle_param(param_idx)
        elif tool_idx == TOOL_CLUSTALO:
            return self._decode_clustalo_param(param_idx)
        else:
            raise ValueError(f"Unknown tool_idx {tool_idx}")

    def describe(self, action_idx: int) -> str:
        """Human-readable description of a flat action."""
        action = self.decode(action_idx)
        return self._describe_action(action)

    def describe_hierarchical(self, tool_idx: int, param_idx: int) -> str:
        """Human-readable description of a hierarchical action."""
        action = self.decode_hierarchical(tool_idx, param_idx)
        return self._describe_action(action)

    def _describe_action(self, action: MSAAction) -> str:
        """Human-readable description of an MSAAction."""
        if action.tool == "mafft":
            return (
                f"MAFFT --{action.mafft_strategy} "
                f"op={action.mafft_op} ep={action.mafft_ep} "
                f"maxiter={action.mafft_maxiterate}"
            )
        elif action.tool == "muscle":
            return (
                f"MUSCLE -{action.muscle_command} "
                f"perm={action.muscle_perm} perturb={action.muscle_perturb}"
            )
        else:
            return (
                f"ClustalO iter={action.clustalo_iter} "
                f"full={'yes' if action.clustalo_full else 'no'} "
                f"full-iter={'yes' if action.clustalo_full_iter else 'no'} "
                f"kimura={'yes' if action.clustalo_kimura else 'no'}"
            )

    def __len__(self) -> int:
        return self.n_actions
