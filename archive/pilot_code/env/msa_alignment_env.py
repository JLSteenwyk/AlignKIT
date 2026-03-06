"""Contextual bandit environment for MSA tool+parameter selection.

Each episode:
1. reset() samples a random MSA test case and returns state features.
2. step(action) decodes the action, runs MAFFT or MUSCLE via subprocess,
   scores against the BAliBASE reference, and returns the reward.
"""

import os
import subprocess
import tempfile
import time
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

from config_msa import MSAConfig
from data.msa_dataset import MSADataset, MSATestCase
from env.msa_action_space import MSAAction, MSAActionSpace
from env.msa_state_features import extract_msa_features
from scoring.msa_sp_score import msa_sp_score
from scoring.msa_tc_score import msa_tc_score
from scoring.reward import compute_reward


class MSAAlignmentEnv:
    """Single-step (bandit) environment for MSA tool selection.

    State: 8-dimensional feature vector of the MSA test case.
    Action: discrete index into MAFFT/MUSCLE/ClustalO parameter space (400 actions).
    Reward: weighted (SP + TC) / 2 compared to BAliBASE reference.
    """

    def __init__(self, dataset: MSADataset, action_space: MSAActionSpace, config: MSAConfig):
        self.dataset = dataset
        self.action_space = action_space
        self.config = config

        self._current_case: Optional[MSATestCase] = None
        self._current_state: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        """Sample a new MSA test case and return state features.

        Returns:
            State feature vector of shape (8,).
        """
        self._current_case = self.dataset.sample_one()
        self._current_state = extract_msa_features(self._current_case.sequences)
        return self._current_state

    def reset_with_case(self, case: MSATestCase) -> np.ndarray:
        """Reset with a specific test case (for evaluation)."""
        self._current_case = case
        self._current_state = extract_msa_features(case.sequences)
        return self._current_state

    def step(self, action_idx: int) -> Tuple[float, Dict]:
        """Execute MSA alignment with the given flat action and compute reward.

        Args:
            action_idx: Index into the flat discrete action space.

        Returns:
            (reward, info_dict) with SP, TC, timing, and action details.
        """
        action = self.action_space.decode(action_idx)
        return self._step_with_action(action)

    def step_hierarchical(self, tool_idx: int, param_idx: int) -> Tuple[float, Dict]:
        """Execute MSA alignment with a hierarchical (tool, param) action.

        Args:
            tool_idx: 0=MAFFT, 1=MUSCLE, 2=ClustalO.
            param_idx: Index within that tool's parameter space.

        Returns:
            (reward, info_dict) with SP, TC, timing, and action details.
        """
        action = self.action_space.decode_hierarchical(tool_idx, param_idx)
        return self._step_with_action(action)

    def _step_with_action(self, action: MSAAction) -> Tuple[float, Dict]:
        """Execute MSA alignment with a decoded action and compute reward."""
        assert self._current_case is not None, "Must call reset() before step()"

        case = self._current_case
        start_time = time.time()

        try:
            if action.tool == "mafft":
                pred_msa = self._run_mafft(case.fasta_path, action)
            elif action.tool == "muscle":
                pred_msa = self._run_muscle(case.fasta_path, action)
            else:
                pred_msa = self._run_clustalo(case.fasta_path, action)

            elapsed = time.time() - start_time

            if pred_msa is None or len(pred_msa) < 2:
                return 0.0, self._make_info(action, case, 0.0, 0.0, 0.0, elapsed, error="empty_alignment")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return 0.0, self._make_info(action, case, 0.0, 0.0, 0.0, elapsed, error="timeout")
        except Exception as e:
            elapsed = time.time() - start_time
            return 0.0, self._make_info(action, case, 0.0, 0.0, 0.0, elapsed, error=str(e))

        # Score against reference
        sp = msa_sp_score(pred_msa, case.ref_alignment)
        tc = msa_tc_score(pred_msa, case.ref_alignment)
        reward = compute_reward(sp, tc, self.config.sp_weight, self.config.tc_weight)

        info = self._make_info(action, case, sp, tc, reward, elapsed)
        return reward, info

    def _run_mafft(self, fasta_path, action: MSAAction) -> Optional[MultipleSeqAlignment]:
        """Run MAFFT and return parsed MSA."""
        cmd = [str(self.config.mafft_bin)]

        # Strategy flag
        if action.mafft_strategy == "localpair":
            cmd.append("--localpair")
        elif action.mafft_strategy == "globalpair":
            cmd.append("--globalpair")
        elif action.mafft_strategy == "genafpair":
            cmd.append("--genafpair")
        # "auto" → no flag (MAFFT auto-selects)

        # Parameters
        cmd.extend(["--op", str(action.mafft_op)])
        cmd.extend(["--ep", str(action.mafft_ep)])
        cmd.extend(["--maxiterate", str(action.mafft_maxiterate)])
        cmd.extend(["--thread", str(self.config.mafft_threads)])
        cmd.append("--quiet")
        cmd.append(str(fasta_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.subprocess_timeout,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        try:
            return AlignIO.read(StringIO(result.stdout), "fasta")
        except Exception:
            return None

    def _run_muscle(self, fasta_path, action: MSAAction) -> Optional[MultipleSeqAlignment]:
        """Run MUSCLE v5 and return parsed MSA."""
        # MUSCLE v5 writes to output file
        with tempfile.NamedTemporaryFile(suffix=".afa", delete=False, mode="w") as tmp:
            output_path = tmp.name

        try:
            cmd = [str(self.config.muscle_bin)]

            # Command mode
            if action.muscle_command == "super5":
                cmd.extend(["-super5", str(fasta_path)])
            else:
                cmd.extend(["-align", str(fasta_path)])

            cmd.extend(["-output", output_path])

            # Permutation
            if action.muscle_perm != "none":
                cmd.extend(["-perm", action.muscle_perm])

            # Perturbation
            if action.muscle_perturb > 0:
                cmd.extend(["-perturb", str(action.muscle_perturb)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.subprocess_timeout,
            )

            if result.returncode != 0:
                return None

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return None

            return AlignIO.read(output_path, "fasta")

        except Exception:
            return None
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _run_clustalo(self, fasta_path, action: MSAAction) -> Optional[MultipleSeqAlignment]:
        """Run Clustal Omega and return parsed MSA."""
        cmd = [str(self.config.clustalo_bin), "-i", str(fasta_path)]

        if action.clustalo_iter > 0:
            cmd.extend(["--iter", str(action.clustalo_iter)])
        if action.clustalo_full:
            cmd.append("--full")
        if action.clustalo_full_iter:
            cmd.append("--full-iter")
        if action.clustalo_kimura:
            cmd.append("--use-kimura")

        cmd.extend(["--threads", str(self.config.mafft_threads)])
        cmd.append("--force")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.subprocess_timeout,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        try:
            return AlignIO.read(StringIO(result.stdout), "fasta")
        except Exception:
            return None

    def _make_info(
        self, action: MSAAction, case: MSATestCase,
        sp: float, tc: float, reward: float, elapsed: float,
        error: Optional[str] = None,
    ) -> Dict:
        """Build info dict for step() return."""
        info = {
            "sp": sp,
            "tc": tc,
            "reward": reward,
            "tool": action.tool,
            "elapsed": elapsed,
            "case_id": case.case_id,
            "ref_set": case.ref_set,
            "num_sequences": case.num_sequences,
        }

        if action.tool == "mafft":
            info.update({
                "mafft_op": action.mafft_op,
                "mafft_ep": action.mafft_ep,
                "mafft_strategy": action.mafft_strategy,
                "mafft_maxiterate": action.mafft_maxiterate,
            })
        elif action.tool == "muscle":
            info.update({
                "muscle_command": action.muscle_command,
                "muscle_perm": action.muscle_perm,
                "muscle_perturb": action.muscle_perturb,
            })
        else:
            info.update({
                "clustalo_iter": action.clustalo_iter,
                "clustalo_full": action.clustalo_full,
                "clustalo_full_iter": action.clustalo_full_iter,
                "clustalo_kimura": action.clustalo_kimura,
            })

        if error is not None:
            info["error"] = error

        return info
