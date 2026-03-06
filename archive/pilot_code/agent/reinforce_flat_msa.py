"""Flat REINFORCE agent for MSA alignment with value baseline."""

from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent.policy_network import PolicyNetwork
from config_msa import MSAConfig


class ReinforceFlatMSAAgent:
    """Flat REINFORCE agent for MSA tool-parameter selection.

    Uses a learned value baseline (shared trunk) with advantage normalization,
    single entropy bonus, and gradient clipping.
    """

    def __init__(self, policy: PolicyNetwork, config: MSAConfig, device: torch.device):
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        # Batch mean baseline for logging / checkpoint compatibility
        self._baseline = 0.0

    def select_action(self, state: np.ndarray) -> tuple:
        """Sample an action from the policy (no gradient tracking).

        Returns:
            (action_idx, log_prob_value, entropy_value)
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.policy.get_action(state_t)
        return action, log_prob.item(), entropy.item()

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Select the greedy (argmax) action.

        Returns:
            action_idx
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.policy(state_t)
            return logits.argmax().item()

    def update(self, batch: Dict[str, List]) -> Dict[str, float]:
        """REINFORCE update with value baseline on a batch of transitions.

        Args:
            batch: dict with keys 'states', 'actions', 'rewards'.

        Returns:
            dict with training metrics.
        """
        states = torch.tensor(
            np.array(batch["states"]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            batch["actions"], dtype=torch.long, device=self.device
        )
        rewards_np = np.array(batch["rewards"], dtype=np.float64)
        rewards_t = torch.tensor(
            rewards_np, dtype=torch.float32, device=self.device
        )

        self._baseline = float(rewards_np.mean())

        # Forward pass with gradients
        log_probs, entropies, values = self.policy.evaluate_actions(states, actions)

        # Value-based advantages with normalization
        advantages_t = rewards_t - values.detach()
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

        # Policy gradient loss
        policy_loss = -(log_probs * advantages_t).mean()

        # Entropy bonus
        entropy_loss = -self.config.entropy_coeff * entropies.mean()

        # Value loss
        value_loss = self.config.value_coeff * torch.nn.functional.mse_loss(
            values, rewards_t
        )

        loss = policy_loss + entropy_loss + value_loss

        # Backward + clip + step
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_max_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
            "advantage_mean": advantages_t.mean().item(),
            "advantage_std": advantages_t.std().item(),
            "baseline": self._baseline,
        }
