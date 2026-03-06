"""Contextual bandit environment for pairwise alignment.

Each episode:
  1. Sample a pair from the dataset
  2. Return features as state
  3. Agent provides gap penalties (continuous action)
  4. Run NW alignment with those penalties
  5. Score against reference alignment
  6. Return reward
"""

import numpy as np

from config import Config
from data.pairwise_dataset import PairwiseDataset, PairwiseExample
from env.needleman_wunsch import needleman_wunsch
from scoring.sp_score import sp_score
from scoring.tc_score import tc_score
from scoring.reward import compute_reward


class PairwiseEnv:
    """Contextual bandit environment for pairwise alignment."""

    def __init__(self, dataset: PairwiseDataset, config: Config):
        self.dataset = dataset
        self.config = config
        self.current_example: PairwiseExample = None

    def reset(self) -> np.ndarray:
        """Sample a new pair and return its features as the state.

        Returns:
            Feature vector, shape (input_dim,).
        """
        self.current_example = self.dataset.sample_one()
        return self.current_example.features

    def reset_with(self, example: PairwiseExample) -> np.ndarray:
        """Reset with a specific example (for eval).

        Returns:
            Feature vector, shape (input_dim,).
        """
        self.current_example = example
        return example.features

    def step(self, action: np.ndarray) -> tuple:
        """Execute alignment with the given gap penalties and compute reward.

        Args:
            action: shape (action_dim,) — first K are gap_open, next K are gap_extend.

        Returns:
            (reward, info_dict)
        """
        K = self.config.num_regions
        gap_open = action[:K].astype(np.float64)
        gap_extend = action[K:].astype(np.float64)

        ex = self.current_example

        # Run NW alignment
        pred_seq1, pred_seq2, nw_score = needleman_wunsch(
            ex.seq1_idx, ex.seq2_idx, gap_open, gap_extend, K
        )

        # Score against reference
        sp = sp_score(pred_seq1, pred_seq2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        tc = tc_score(pred_seq1, pred_seq2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        reward = compute_reward(sp, tc, self.config.sp_weight, self.config.tc_weight)

        info = {
            "sp": sp,
            "tc": tc,
            "nw_score": nw_score,
            "gap_open": gap_open.tolist(),
            "gap_extend": gap_extend.tolist(),
            "ref_set": ex.ref_set,
            "pair_id": ex.pair_id,
            "seq1_len": len(ex.seq1_raw),
            "seq2_len": len(ex.seq2_raw),
        }

        return reward, info
