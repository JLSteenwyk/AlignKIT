"""Gaussian policy with sigmoid transform for continuous gap penalty prediction.

Architecture:
  Input (12) -> [128] -> ReLU -> [128] -> ReLU -> shared trunk
    -> mean_head -> 2K raw means
    -> value_head -> 1 scalar
    + log_std: nn.Parameter(2K,) -- learned, state-independent

Action transform: sigmoid(raw) * (max - min) + min
  gap_open:   [1.0, 20.0]
  gap_extend: [0.1, 5.0]

Log-prob correction: subtract log(sigmoid'(x) * range) = log(sig(x)*(1-sig(x))*range)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ContinuousPolicy(nn.Module):
    """Gaussian policy that outputs position-dependent gap penalties."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_sizes: List[int] = None,
        action_dim: int = 6,
        initial_log_std: float = 0.0,
        min_std: float = 0.05,
        gap_open_range: Tuple[float, float] = (1.0, 20.0),
        gap_extend_range: Tuple[float, float] = (0.1, 5.0),
        num_regions: int = 3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.action_dim = action_dim
        self.min_log_std = float(np.log(min_std))
        self.num_regions = num_regions

        # Action bounds: first K dims are gap_open, next K are gap_extend
        K = num_regions
        lo = [gap_open_range[0]] * K + [gap_extend_range[0]] * K
        hi = [gap_open_range[1]] * K + [gap_extend_range[1]] * K
        self.register_buffer("action_lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("action_hi", torch.tensor(hi, dtype=torch.float32))
        self.register_buffer(
            "action_range", torch.tensor([h - l for h, l in zip(hi, lo)], dtype=torch.float32)
        )

        # Shared trunk
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.trunk = nn.Sequential(*layers)

        # Heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)

        # Learned log-std (state-independent)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def _get_std(self) -> torch.Tensor:
        """Clamp log_std and return std."""
        return torch.exp(torch.clamp(self.log_std, min=self.min_log_std))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: state -> (raw_mean, value).

        Args:
            state: shape (batch, input_dim) or (input_dim,).

        Returns:
            raw_mean: shape (batch, action_dim) or (action_dim,)
            value: shape (batch,) or scalar
        """
        features = self.trunk(state)
        raw_mean = self.mean_head(features)
        value = self.value_head(features).squeeze(-1)
        return raw_mean, value

    def transform_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Transform raw action through sigmoid into bounded action space.

        Args:
            raw_action: unbounded, shape (..., action_dim)

        Returns:
            Transformed action in [lo, hi], shape (..., action_dim)
        """
        return torch.sigmoid(raw_action) * self.action_range + self.action_lo

    def log_prob_correction(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Compute log-prob correction for the sigmoid transform.

        The correction is: -sum(log(sigmoid'(x) * range))
        where sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        Args:
            raw_action: shape (..., action_dim)

        Returns:
            Correction scalar per sample, shape (...,)
        """
        sig = torch.sigmoid(raw_action)
        # log(sig * (1-sig) * range) for each dim, then sum
        log_det = torch.log(sig * (1 - sig) * self.action_range + 1e-8)
        return log_det.sum(dim=-1)

    def get_action(self, state: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            state: shape (input_dim,) — single state.

        Returns:
            (transformed_action_np, raw_action, log_prob, entropy)
            raw_action is stored for log-prob recomputation during update.
        """
        raw_mean, _ = self.forward(state)
        std = self._get_std()
        dist = Normal(raw_mean, std)
        raw_action = dist.sample()

        # Log prob in raw space
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        # Correction for sigmoid transform
        correction = self.log_prob_correction(raw_action)
        log_prob = log_prob_raw - correction

        # Entropy (approximate: raw-space entropy, ignoring transform)
        entropy = dist.entropy().sum(dim=-1)

        # Transform to bounded action
        action = self.transform_action(raw_action)

        return action.detach().cpu().numpy(), raw_action.detach(), log_prob.detach(), entropy.detach()

    def evaluate_actions(
        self, states: torch.Tensor, raw_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-probs, entropies, and values for a batch.

        Args:
            states: shape (batch, input_dim)
            raw_actions: shape (batch, action_dim) — pre-transform actions

        Returns:
            (log_probs, entropies, values) each shape (batch,)
        """
        raw_mean, values = self.forward(states)
        std = self._get_std()
        dist = Normal(raw_mean, std)

        log_prob_raw = dist.log_prob(raw_actions).sum(dim=-1)
        correction = self.log_prob_correction(raw_actions)
        log_probs = log_prob_raw - correction

        entropies = dist.entropy().sum(dim=-1)

        return log_probs, entropies, values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate."""
        features = self.trunk(state)
        return self.value_head(features).squeeze(-1)

    def get_greedy_action(self, state: torch.Tensor) -> np.ndarray:
        """Return the mean (greedy) action, transformed to bounded space."""
        raw_mean, _ = self.forward(state)
        action = self.transform_action(raw_mean)
        return action.detach().cpu().numpy()
