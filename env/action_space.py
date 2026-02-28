"""Discrete action space: encode/decode action index to alignment parameters."""

from typing import Tuple

import numpy as np

from config import Config


class ActionSpace:
    """Maps a flat action index to (gap_open, gap_extend, matrix_name) and back.

    Total actions = gap_open_bins * gap_extend_bins * num_matrices.
    Index layout: action = i * (gap_extend_bins * num_matrices) + j * num_matrices + k
    where i=gap_open bin, j=gap_extend bin, k=matrix index.
    """

    def __init__(self, config: Config):
        self.gap_open_values = np.linspace(
            config.gap_open_min, config.gap_open_max, config.gap_open_bins
        )
        self.gap_extend_values = np.linspace(
            config.gap_extend_min, config.gap_extend_max, config.gap_extend_bins
        )
        self.matrices = list(config.matrices)

        self.n_gap_open = len(self.gap_open_values)
        self.n_gap_extend = len(self.gap_extend_values)
        self.n_matrices = len(self.matrices)
        self.n_actions = self.n_gap_open * self.n_gap_extend * self.n_matrices

    def decode(self, action_idx: int) -> Tuple[float, float, str]:
        """Decode flat action index to (gap_open, gap_extend, matrix_name)."""
        assert 0 <= action_idx < self.n_actions, f"Action {action_idx} out of range [0, {self.n_actions})"

        k = action_idx % self.n_matrices
        remainder = action_idx // self.n_matrices
        j = remainder % self.n_gap_extend
        i = remainder // self.n_gap_extend

        return (
            float(self.gap_open_values[i]),
            float(self.gap_extend_values[j]),
            self.matrices[k],
        )

    def encode(self, i: int, j: int, k: int) -> int:
        """Encode (gap_open_bin, gap_extend_bin, matrix_idx) to flat index."""
        return i * (self.n_gap_extend * self.n_matrices) + j * self.n_matrices + k

    def __len__(self) -> int:
        return self.n_actions
