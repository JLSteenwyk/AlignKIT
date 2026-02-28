"""Main training loop for the REINFORCE alignment agent."""

import random
import time

import numpy as np
import torch

from agent.policy_network import PolicyNetwork
from agent.reinforce import ReinforceAgent
from config import Config
from data.dataset import build_datasets
from env.action_space import ActionSpace
from env.alignment_env import AlignmentBanditEnv
from training.checkpointer import save_checkpoint
from training.logger import JSONLLogger


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: Config):
    """Run the full REINFORCE training loop.

    1. Load BAliBASE data, build datasets
    2. Create environment, agent
    3. Collect batches of episodes, update policy
    4. Log metrics, checkpoint, evaluate periodically
    """
    config.ensure_dirs()
    set_seeds(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataset, eval_dataset, _ = build_datasets(
        config.data_dir, config.train_ref_sets, config.eval_ref_sets, config.max_seq_length
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training data loaded. Check BAliBASE data path.")

    # Environment + action space
    action_space = ActionSpace(config)
    env = AlignmentBanditEnv(train_dataset, action_space, config)
    eval_env = AlignmentBanditEnv(eval_dataset, action_space, config)

    # Agent
    policy = PolicyNetwork(
        input_dim=config.input_dim,
        hidden_sizes=config.hidden_sizes,
        output_dim=config.num_actions,
    )
    agent = ReinforceAgent(policy, config, device)

    # Logging
    train_logger = JSONLLogger(config.log_dir, "train")
    eval_logger = JSONLLogger(config.log_dir, "eval")

    print(f"\nStarting training: {config.num_episodes} episodes, batch_size={config.batch_size}")
    print(f"Action space: {config.num_actions} actions")
    print(f"Train pairs: {len(train_dataset)}, Eval pairs: {len(eval_dataset)}")
    print()

    # Training loop
    episode = 0
    total_start = time.time()

    # Running stats for logging
    recent_rewards = []
    recent_sp = []
    recent_tc = []

    while episode < config.num_episodes:
        # Collect a batch of episodes
        batch = {"states": [], "actions": [], "rewards": []}
        batch_infos = []

        for _ in range(config.batch_size):
            state = env.reset()
            action, log_prob, entropy = agent.select_action(state)
            reward, info = env.step(action)

            batch["states"].append(state)
            batch["actions"].append(action)
            batch["rewards"].append(reward)
            batch_infos.append(info)

            recent_rewards.append(reward)
            recent_sp.append(info.get("sp", 0.0))
            recent_tc.append(info.get("tc", 0.0))

        # Update policy
        update_metrics = agent.update(batch)
        episode += config.batch_size

        # Log
        if episode % config.log_every < config.batch_size:
            avg_reward = np.mean(recent_rewards[-config.log_every:])
            avg_sp = np.mean(recent_sp[-config.log_every:])
            avg_tc = np.mean(recent_tc[-config.log_every:])

            metrics = {
                "reward_mean": float(avg_reward),
                "sp_mean": float(avg_sp),
                "tc_mean": float(avg_tc),
                **update_metrics,
            }
            train_logger.log(metrics, episode)

            elapsed = time.time() - total_start
            print(
                f"Ep {episode:6d} | "
                f"R={avg_reward:.4f} SP={avg_sp:.4f} TC={avg_tc:.4f} | "
                f"Ent={update_metrics['entropy']:.3f} "
                f"Loss={update_metrics['loss']:.4f} | "
                f"{elapsed:.0f}s"
            )

        # Checkpoint
        if episode % config.checkpoint_every < config.batch_size:
            save_checkpoint(policy, agent.optimizer, episode, config.checkpoint_dir)

        # Evaluate
        if episode % config.eval_every < config.batch_size and len(eval_dataset) > 0:
            eval_metrics = _evaluate(agent, eval_env, eval_dataset, config)
            eval_logger.log(eval_metrics, episode)
            print(
                f"  [EVAL] R={eval_metrics['reward_mean']:.4f} "
                f"SP={eval_metrics['sp_mean']:.4f} TC={eval_metrics['tc_mean']:.4f} "
                f"(n={eval_metrics['n_pairs']})"
            )

    # Final save
    save_checkpoint(policy, agent.optimizer, episode, config.checkpoint_dir, "checkpoint_final.pt")
    total_time = time.time() - total_start
    print(f"\nTraining complete: {episode} episodes in {total_time:.1f}s")


def _evaluate(agent, env, dataset, config, max_pairs=200):
    """Run greedy evaluation on the eval dataset."""
    rewards, sps, tcs = [], [], []
    per_set = {}

    pairs = list(dataset)
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    for ref in pairs:
        state = env.reset_with_ref(ref)
        action = agent.select_action_greedy(state)
        reward, info = env.step(action)

        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))

        rs = info.get("ref_set", "unknown")
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info.get("sp", 0.0))
        per_set[rs]["tcs"].append(info.get("tc", 0.0))

    result = {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "sp_mean": float(np.mean(sps)) if sps else 0.0,
        "tc_mean": float(np.mean(tcs)) if tcs else 0.0,
        "n_pairs": len(rewards),
    }

    for rs, data in per_set.items():
        result[f"{rs}_reward"] = float(np.mean(data["rewards"]))
        result[f"{rs}_sp"] = float(np.mean(data["sps"]))
        result[f"{rs}_tc"] = float(np.mean(data["tcs"]))

    return result
