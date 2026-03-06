"""Contextual bandit environment for MSA with continuous MAFFT parameters.

Each episode:
  1. Sample an MSA test case, return 20-dim features as state
  2. Agent predicts (op, ep) for MAFFT
  3. Run MAFFT --localpair --maxiterate 10 with those params
  4. Score against BAliBASE reference
  5. Return reward
"""

import subprocess
import time
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

from data.msa_dataset import MSADataset, MSATestCase
from env.msa_features import extract_msa_features
from scoring.msa_sp_score import msa_sp_score
from scoring.msa_tc_score import msa_tc_score
from scoring.reward import compute_reward


class MSAEnv:
    """Contextual bandit environment for MAFFT parameter prediction."""

    def __init__(self, dataset: MSADataset, config):
        self.dataset = dataset
        self.config = config
        self._current_case: Optional[MSATestCase] = None

    def reset(self) -> np.ndarray:
        self._current_case = self.dataset.sample_one()
        return extract_msa_features(self._current_case.sequences)

    def reset_with(self, case: MSATestCase) -> np.ndarray:
        self._current_case = case
        return extract_msa_features(case.sequences)

    def step(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Run MAFFT with predicted (op, ep) and score against reference.

        Args:
            action: shape (2,) — [op, ep]

        Returns:
            (reward, info_dict)
        """
        op = float(action[0])
        ep = float(action[1])
        case = self._current_case

        start_time = time.time()

        try:
            pred_msa = self._run_mafft(case.fasta_path, op, ep)
            elapsed = time.time() - start_time

            if pred_msa is None or len(pred_msa) < 2:
                return 0.0, self._info(case, op, ep, 0.0, 0.0, 0.0, elapsed, "empty_alignment")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return 0.0, self._info(case, op, ep, 0.0, 0.0, 0.0, elapsed, "timeout")
        except Exception as e:
            elapsed = time.time() - start_time
            return 0.0, self._info(case, op, ep, 0.0, 0.0, 0.0, elapsed, str(e))

        sp = msa_sp_score(pred_msa, case.ref_alignment)
        tc = msa_tc_score(pred_msa, case.ref_alignment)
        reward = compute_reward(sp, tc, self.config.msa_sp_weight, self.config.msa_tc_weight)

        return reward, self._info(case, op, ep, sp, tc, reward, elapsed)

    def _run_mafft(self, fasta_path: Path, op: float, ep: float) -> Optional[MultipleSeqAlignment]:
        cmd = [
            str(self.config.mafft_bin),
            "--localpair",
            "--maxiterate", "10",
            "--op", str(op),
            "--ep", str(ep),
            "--thread", str(self.config.mafft_threads),
            "--quiet",
            str(fasta_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=self.config.mafft_timeout,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        try:
            return AlignIO.read(StringIO(result.stdout), "fasta")
        except Exception:
            return None

    def _info(self, case, op, ep, sp, tc, reward, elapsed, error=None):
        info = {
            "sp": sp, "tc": tc, "reward": reward,
            "op": op, "ep": ep,
            "elapsed": elapsed,
            "case_id": case.case_id,
            "ref_set": case.ref_set,
            "num_sequences": case.num_sequences,
        }
        if error:
            info["error"] = error
        return info
