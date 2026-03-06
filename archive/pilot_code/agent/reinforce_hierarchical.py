"""REINFORCE agent with hierarchical (tool + params) policy."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent.hierarchical_policy import HierarchicalPolicy
from config_msa import MSAConfig


class HierarchicalReinforceAgent:
    """REINFORCE agent wrapping a HierarchicalPolicy.

    Decomposed entropy bonus:
        -tool_entropy_coeff * H(tool) - param_entropy_coeff * H(params|tool)
    """

    def __init__(
        self, policy: HierarchicalPolicy, config: MSAConfig, device: torch.device
    ):
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        # Batch mean baseline for logging compatibility
        self._baseline = 0.0

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, int, float, float, float]:
        """Select action using current policy (no gradient tracking).

        Returns:
            (tool_idx, param_idx, log_prob, tool_entropy, param_entropy)
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            (tool_idx, param_idx), log_prob, tool_ent, param_ent = (
                self.policy.get_action(state_t)
            )
        return tool_idx, param_idx, log_prob.item(), tool_ent.item(), param_ent.item()

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        """Select greedy action via argmax at both levels.

        Returns:
            (tool_idx, param_idx)
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            return self.policy.get_action_greedy(state_t)

    def update(self, batch: Dict[str, List]) -> Dict[str, float]:
        """Perform a REINFORCE update on a batch of hierarchical transitions.

        Args:
            batch: dict with keys:
                - states: list of numpy arrays, each shape (input_dim,)
                - tool_indices: list of int
                - param_indices: list of int
                - rewards: list of float

        Returns:
            dict with training metrics.
        """
        states = torch.tensor(
            np.array(batch["states"]), dtype=torch.float32, device=self.device
        )
        tool_indices = torch.tensor(
            batch["tool_indices"], dtype=torch.long, device=self.device
        )
        param_indices = torch.tensor(
            batch["param_indices"], dtype=torch.long, device=self.device
        )
        rewards_np = np.array(batch["rewards"], dtype=np.float64)
        rewards_t = torch.tensor(
            rewards_np, dtype=torch.float32, device=self.device
        )

        # Update batch mean baseline for logging
        self._baseline = float(rewards_np.mean())

        # Forward pass with gradients (includes value predictions)
        log_probs, tool_entropies, param_entropies, values = (
            self.policy.evaluate_actions(states, tool_indices, param_indices)
        )

        # Value-based advantages with normalization
        advantages_t = rewards_t - values.detach()
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

        # Policy gradient loss
        policy_loss = -(log_probs * advantages_t).mean()

        # Decomposed entropy bonus
        tool_entropy_loss = -self.config.tool_entropy_coeff * tool_entropies.mean()
        param_entropy_loss = -self.config.param_entropy_coeff * param_entropies.mean()

        # Value loss
        value_loss = self.config.value_coeff * torch.nn.functional.mse_loss(
            values, rewards_t
        )

        loss = policy_loss + tool_entropy_loss + param_entropy_loss + value_loss

        # Backward + clip + step
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_max_norm)
        self.optimizer.step()

        tool_ent_val = tool_entropies.mean().item()
        param_ent_val = param_entropies.mean().item()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "tool_entropy": tool_ent_val,
            "param_entropy": param_ent_val,
            "entropy": tool_ent_val + param_ent_val,  # backward compat
            "advantage_mean": advantages_t.mean().item(),
            "advantage_std": advantages_t.std().item(),
            "baseline": self._baseline,
        }
