"""Contextual bandit environment for pairwise sequence alignment.

Each episode:
1. reset() samples a random sequence pair and returns state features.
2. step(action) decodes the action into alignment parameters, runs BioPython
   PairwiseAligner, scores against the reference, and returns the reward.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

from config import Config
from data.balibase_parser import PairwiseReference
from data.dataset import PairwiseDataset
from env.action_space import ActionSpace
from env.state_features import extract_features
from scoring.sp_score import sp_score
from scoring.tc_score import tc_score
from scoring.reward import compute_reward


class AlignmentBanditEnv:
    """Single-step (bandit) environment for learning alignment parameters.

    State: 5-dimensional feature vector of the sequence pair.
    Action: discrete index into (gap_open, gap_extend, matrix) space.
    Reward: (SP + TC) / 2 compared to BAliBASE reference.
    """

    def __init__(self, dataset: PairwiseDataset, action_space: ActionSpace, config: Config):
        self.dataset = dataset
        self.action_space = action_space
        self.config = config

        # Pre-load substitution matrices
        self._matrices = {}
        for name in config.matrices:
            self._matrices[name] = substitution_matrices.load(name)

        self._current_ref: Optional[PairwiseReference] = None
        self._current_state: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        """Sample a new sequence pair and return state features.

        Returns:
            State feature vector of shape (5,).
        """
        self._current_ref = self.dataset.sample_one()
        self._current_state = extract_features(
            self._current_ref.seq1_raw, self._current_ref.seq2_raw
        )
        return self._current_state

    def reset_with_ref(self, ref: PairwiseReference) -> np.ndarray:
        """Reset with a specific reference pair (for evaluation)."""
        self._current_ref = ref
        self._current_state = extract_features(ref.seq1_raw, ref.seq2_raw)
        return self._current_state

    def step(self, action_idx: int) -> Tuple[float, Dict]:
        """Execute alignment with the given action and compute reward.

        Args:
            action_idx: Index into the discrete action space.

        Returns:
            (reward, info_dict) where info_dict contains SP, TC, and action details.
        """
        assert self._current_ref is not None, "Must call reset() before step()"

        ref = self._current_ref
        gap_open, gap_extend, matrix_name = self.action_space.decode(action_idx)

        # Configure aligner
        aligner = Align.PairwiseAligner()
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend
        aligner.substitution_matrix = self._matrices[matrix_name]

        # Run alignment
        try:
            alignments = aligner.align(ref.seq1_raw, ref.seq2_raw)
            alignment = alignments[0]

            # Extract aligned sequences
            pred_seq1, pred_seq2 = self._extract_aligned_sequences(alignment, ref)
        except Exception as e:
            # If alignment fails, return zero reward
            return 0.0, {
                "sp": 0.0, "tc": 0.0, "reward": 0.0,
                "gap_open": gap_open, "gap_extend": gap_extend,
                "matrix": matrix_name, "error": str(e),
                "ref_set": ref.ref_set,
            }

        # Score against reference
        sp = sp_score(pred_seq1, pred_seq2, ref.seq1_aligned, ref.seq2_aligned)
        tc = tc_score(pred_seq1, pred_seq2, ref.seq1_aligned, ref.seq2_aligned)
        reward = compute_reward(sp, tc, self.config.sp_weight, self.config.tc_weight)

        info = {
            "sp": sp,
            "tc": tc,
            "reward": reward,
            "gap_open": gap_open,
            "gap_extend": gap_extend,
            "matrix": matrix_name,
            "ref_set": ref.ref_set,
            "seq1_len": len(ref.seq1_raw),
            "seq2_len": len(ref.seq2_raw),
        }

        return reward, info

    def _extract_aligned_sequences(
        self, alignment, ref: PairwiseReference
    ) -> Tuple[str, str]:
        """Extract aligned sequence strings from a BioPython alignment object.

        Uses the alignment's coordinates to reconstruct gapped sequences.
        """
        try:
            # Try format-based extraction first
            formatted = format(alignment)
            lines = formatted.strip().split("\n")

            # BioPython format: alternating target/query lines with possible
            # separator lines. Extract lines that contain sequence characters.
            seq_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("|") and not all(
                    c in ".|: " for c in stripped
                ):
                    seq_lines.append(stripped)

            if len(seq_lines) >= 2:
                # Parse aligned sequences from formatted output
                return self._parse_formatted_alignment(alignment, ref)
        except Exception:
            pass

        # Fallback: reconstruct from coordinates
        return self._reconstruct_from_coordinates(alignment, ref)

    def _parse_formatted_alignment(
        self, alignment, ref: PairwiseReference
    ) -> Tuple[str, str]:
        """Reconstruct aligned sequences from alignment coordinates."""
        return self._reconstruct_from_coordinates(alignment, ref)

    def _reconstruct_from_coordinates(
        self, alignment, ref: PairwiseReference
    ) -> Tuple[str, str]:
        """Reconstruct aligned sequences using alignment coordinates.

        BioPython PairwiseAligner stores alignments as coordinate arrays.
        We reconstruct gapped sequences from these coordinates.
        """
        coords = alignment.coordinates
        seq1 = ref.seq1_raw
        seq2 = ref.seq2_raw

        aligned1 = []
        aligned2 = []

        for seg_idx in range(coords.shape[1] - 1):
            start1, end1 = coords[0, seg_idx], coords[0, seg_idx + 1]
            start2, end2 = coords[1, seg_idx], coords[1, seg_idx + 1]

            len1 = abs(end1 - start1)
            len2 = abs(end2 - start2)

            if len1 > 0 and len2 > 0:
                # Match/mismatch region
                s1 = min(start1, end1)
                s2 = min(start2, end2)
                for k in range(len1):
                    aligned1.append(seq1[s1 + k])
                    aligned2.append(seq2[s2 + k])
            elif len1 > 0:
                # Gap in seq2
                s1 = min(start1, end1)
                for k in range(len1):
                    aligned1.append(seq1[s1 + k])
                    aligned2.append("-")
            elif len2 > 0:
                # Gap in seq1
                s2 = min(start2, end2)
                for k in range(len2):
                    aligned1.append("-")
                    aligned2.append(seq2[s2 + k])

        return "".join(aligned1), "".join(aligned2)
