"""REINFORCE agent with EMA baseline and entropy bonus."""

from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent.policy_network import PolicyNetwork
from config import Config


class ReinforceAgent:
    """REINFORCE policy gradient agent.

    Uses:
    - Exponential moving average (EMA) baseline for variance reduction
    - Entropy bonus to encourage exploration
    - Gradient clipping for stability
    """

    def __init__(self, policy: PolicyNetwork, config: Config, device: torch.device):
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        # EMA baseline
        self._baseline = 0.0
        self._baseline_initialized = False

    def select_action(self, state: np.ndarray) -> tuple:
        """Select action using the current policy (no gradient tracking).

        Args:
            state: numpy array of shape (5,).

        Returns:
            (action_idx, log_prob_value, entropy_value)
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.policy(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob.item(), entropy.item()

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Select the greedy (highest probability) action.

        Args:
            state: numpy array of shape (5,).

        Returns:
            action_idx
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.policy(state_t)
            return logits.argmax().item()

    def update(self, batch: Dict[str, List]) -> Dict[str, float]:
        """Perform a REINFORCE update on a batch of transitions.

        Args:
            batch: dict with keys 'states', 'actions', 'rewards'
                - states: list of numpy arrays, each shape (5,)
                - actions: list of int
                - rewards: list of float

        Returns:
            dict with training metrics: loss, entropy, advantage_mean, baseline
        """
        states = torch.tensor(np.array(batch["states"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = np.array(batch["rewards"], dtype=np.float64)

        # Update EMA baseline
        batch_mean_reward = rewards.mean()
        if not self._baseline_initialized:
            self._baseline = batch_mean_reward
            self._baseline_initialized = True
        else:
            self._baseline = (
                self.config.baseline_momentum * self._baseline
                + (1 - self.config.baseline_momentum) * batch_mean_reward
            )

        # Compute advantages
        advantages = rewards - self._baseline
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Forward pass with gradients
        log_probs, entropies = self.policy.evaluate_actions(states, actions)

        # Policy gradient loss: -E[log_prob * advantage]
        policy_loss = -(log_probs * advantages_t).mean()

        # Entropy bonus (subtract because we minimize loss)
        entropy_loss = -self.config.entropy_coeff * entropies.mean()

        loss = policy_loss + entropy_loss

        # Backward + clip + step
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_max_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropies.mean().item(),
            "advantage_mean": advantages.mean(),
            "advantage_std": advantages.std(),
            "baseline": self._baseline,
        }
