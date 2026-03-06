"""Hierarchical factored policy for MSA tool+parameter selection.

Two-level architecture:
  State -> SharedEncoder -> tool_head (3-way)
                         -> mafft_param_head (N_mafft)
                         -> muscle_param_head (N_muscle)
                         -> clustalo_param_head (N_clustalo)

The tool is sampled first, then parameters are sampled from the
corresponding tool-specific head.  log_prob = log P(tool|s) + log P(params|tool,s).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class HierarchicalPolicy(nn.Module):
    """Factored policy: tool selection + per-tool parameter selection."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        param_sizes: List[int],
    ):
        """
        Args:
            input_dim: Dimension of state features.
            hidden_sizes: Sizes for the shared encoder MLP layers.
            param_sizes: [n_mafft, n_muscle, n_clustalo] parameter counts.
        """
        super().__init__()

        self.param_sizes = param_sizes
        self.n_tools = len(param_sizes)  # 3

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self._encoder_out_dim = prev_dim

        # Tool head: 3-way
        self.tool_head = nn.Linear(prev_dim, self.n_tools)

        # Value head: state -> scalar value estimate
        self.value_head = nn.Linear(prev_dim, 1)

        # Per-tool parameter heads
        self.param_heads = nn.ModuleList([
            nn.Linear(prev_dim, n_params) for n_params in param_sizes
        ])

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass: state -> (tool_logits, [param_logits_per_tool]).

        Args:
            state: (batch, input_dim) or (input_dim,).

        Returns:
            tool_logits: (batch, 3) or (3,).
            param_logits: list of 3 tensors, each (batch, n_params_i) or (n_params_i,).
        """
        h = self.encoder(state)
        tool_logits = self.tool_head(h)
        param_logits = [head(h) for head in self.param_heads]
        return tool_logits, param_logits

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate for a state or batch of states."""
        h = self.encoder(state)
        return self.value_head(h).squeeze(-1)

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy (single state, no batch dim).

        Returns:
            ((tool_idx, param_idx), log_prob, tool_entropy, param_entropy)
            where log_prob = log P(tool|s) + log P(params|tool,s).
        """
        tool_logits, param_logits_list = self.forward(state)

        tool_dist = Categorical(logits=tool_logits)
        tool = tool_dist.sample()
        tool_idx = tool.item()

        param_dist = Categorical(logits=param_logits_list[tool_idx])
        param = param_dist.sample()
        param_idx = param.item()

        log_prob = tool_dist.log_prob(tool) + param_dist.log_prob(param)
        tool_entropy = tool_dist.entropy()
        param_entropy = param_dist.entropy()

        return (tool_idx, param_idx), log_prob, tool_entropy, param_entropy

    def get_action_greedy(self, state: torch.Tensor) -> Tuple[int, int]:
        """Select the greedy action via argmax at both levels."""
        tool_logits, param_logits_list = self.forward(state)
        tool_idx = tool_logits.argmax().item()
        param_idx = param_logits_list[tool_idx].argmax().item()
        return tool_idx, param_idx

    def evaluate_actions(
        self,
        states: torch.Tensor,
        tool_indices: torch.Tensor,
        param_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, entropies, and values for a batch of hierarchical actions.

        Args:
            states: (batch, input_dim).
            tool_indices: (batch,) long tensor with tool index per sample.
            param_indices: (batch,) long tensor with param index per sample.

        Returns:
            (log_probs, tool_entropies, param_entropies, values) each of shape (batch,).
        """
        h = self.encoder(states)
        tool_logits = self.tool_head(h)
        param_logits_list = [head(h) for head in self.param_heads]
        values = self.value_head(h).squeeze(-1)

        batch_size = states.shape[0]

        tool_dist = Categorical(logits=tool_logits)
        tool_log_probs = tool_dist.log_prob(tool_indices)
        tool_entropies = tool_dist.entropy()

        # Param log-probs and entropies: loop over tools with boolean mask
        param_log_probs = torch.zeros(batch_size, device=states.device)
        param_entropies = torch.zeros(batch_size, device=states.device)

        for t in range(self.n_tools):
            mask = tool_indices == t
            if not mask.any():
                continue
            param_dist = Categorical(logits=param_logits_list[t][mask])
            param_log_probs[mask] = param_dist.log_prob(param_indices[mask])
            param_entropies[mask] = param_dist.entropy()

        log_probs = tool_log_probs + param_log_probs
        return log_probs, tool_entropies, param_entropies, values
