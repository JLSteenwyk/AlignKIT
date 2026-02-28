"""Feed-forward policy network for REINFORCE."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple MLP policy: state features -> action logits.

    Architecture: input_dim -> hidden1 -> ReLU -> hidden2 -> ReLU -> output_dim
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_sizes: List[int] = None,
        output_dim: int = 500,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> logits.

        Args:
            state: Tensor of shape (batch, input_dim) or (input_dim,).

        Returns:
            Logits of shape (batch, output_dim) or (output_dim,).
        """
        return self.network(state)

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            state: Tensor of shape (input_dim,) — single state.

        Returns:
            (action_idx, log_prob, entropy)
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropies for a batch of state-action pairs.

        Args:
            states: Tensor of shape (batch, input_dim).
            actions: Tensor of shape (batch,) with action indices.

        Returns:
            (log_probs, entropies) each of shape (batch,).
        """
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, entropies
