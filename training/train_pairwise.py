"""Training loop for pairwise RL gap penalty learning."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from agent.continuous_policy import ContinuousPolicy
from agent.reinforce_continuous import ReinforceContinuousAgent
from config import Config
from data.pairwise_dataset import build_pairwise_datasets
from env.pairwise_env import PairwiseEnv
from training.checkpointer import save_checkpoint, load_checkpoint


def _evaluate(agent, eval_env, eval_dataset, config, greedy=True):
    """Run evaluation on the eval dataset.

    Returns dict of metrics.
    """
    rewards, sps, tcs = [], [], []

    for ex in eval_dataset:
        state = eval_env.reset_with(ex)
        if greedy:
            action = agent.select_action_greedy(state)
        else:
            action, _, _, _ = agent.select_action(state)
        reward, info = eval_env.step(action)
        rewards.append(reward)
        sps.append(info["sp"])
        tcs.append(info["tc"])

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_pairs": len(rewards),
    }


def train_pairwise(config: Config, resume_from: str = None):
    """Main training loop for pairwise alignment RL."""

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Device
    device = torch.device("cpu")

    # Data
    train_dataset, eval_dataset = build_pairwise_datasets(
        config.data_dir, config.train_ref_sets, config.eval_ref_sets,
        config.max_seq_length,
    )

    if len(train_dataset) == 0:
        print("No training data found. Run setup.sh to download BAliBASE.")
        return

    # Environment
    train_env = PairwiseEnv(train_dataset, config)
    eval_env = PairwiseEnv(eval_dataset, config)

    # Policy + Agent
    policy = ContinuousPolicy(
        input_dim=config.input_dim,
        hidden_sizes=config.hidden_sizes,
        action_dim=config.action_dim,
        initial_log_std=config.initial_log_std,
        min_std=config.min_std,
        gap_open_range=(config.gap_open_min, config.gap_open_max),
        gap_extend_range=(config.gap_extend_min, config.gap_extend_max),
        num_regions=config.num_regions,
    )
    agent = ReinforceContinuousAgent(policy, config, device)

    start_episode = 0

    # Resume from checkpoint
    if resume_from:
        ckpt = load_checkpoint(policy, agent.optimizer, config.checkpoint_dir, resume_from, device)
        start_episode = ckpt.get("episode", 0)
        print(f"Resumed from episode {start_episode}")

    # Logging
    config.log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = config.log_dir / "pw_train.jsonl"
    eval_log_path = config.log_dir / "pw_eval.jsonl"

    print(f"\nStarting training: {config.num_episodes} episodes, batch_size={config.batch_size}")
    print(f"Train: {len(train_dataset)} pairs, Eval: {len(eval_dataset)} pairs")
    print(f"Action dim: {config.action_dim} ({config.num_regions} regions x 2)\n")

    episode = start_episode
    batch_count = 0

    while episode < config.num_episodes:
        batch_states = []
        batch_raw_actions = []
        batch_rewards = []
        batch_infos = []
        batch_start = time.time()

        for _ in range(config.batch_size):
            state = train_env.reset()
            action, raw_action, log_prob, entropy = agent.select_action(state)
            reward, info = train_env.step(action)

            batch_states.append(state)
            batch_raw_actions.append(raw_action)
            batch_rewards.append(reward)
            batch_infos.append(info)
            episode += 1

        # Update
        batch = {
            "states": batch_states,
            "raw_actions": batch_raw_actions,
            "rewards": batch_rewards,
        }
        metrics = agent.update(batch)
        batch_elapsed = time.time() - batch_start
        batch_count += 1

        # Log training
        train_record = {
            "episode": episode,
            "batch": batch_count,
            "reward_mean": float(np.mean(batch_rewards)),
            "reward_std": float(np.std(batch_rewards)),
            "sp_mean": float(np.mean([i["sp"] for i in batch_infos])),
            "tc_mean": float(np.mean([i["tc"] for i in batch_infos])),
            "elapsed": batch_elapsed,
            **metrics,
        }
        with open(train_log_path, "a") as f:
            f.write(json.dumps(train_record) + "\n")

        # Print progress
        if batch_count % 10 == 0:
            avg_go = np.mean([i["gap_open"] for i in batch_infos], axis=0)
            avg_ge = np.mean([i["gap_extend"] for i in batch_infos], axis=0)
            print(
                f"[Ep {episode:6d}] R={train_record['reward_mean']:.4f} "
                f"SP={train_record['sp_mean']:.4f} TC={train_record['tc_mean']:.4f} "
                f"loss={metrics['loss']:.4f} ent={metrics['entropy']:.2f} "
                f"go={avg_go.round(1)} ge={avg_ge.round(1)} "
                f"({batch_elapsed:.1f}s)"
            )

        # Checkpoint
        if episode % config.checkpoint_every < config.batch_size:
            save_checkpoint(
                policy, agent.optimizer, episode,
                config.checkpoint_dir,
                extra={"baseline": agent._baseline},
            )

        # Eval
        if episode % config.eval_every < config.batch_size and len(eval_dataset) > 0:
            eval_start = time.time()
            eval_metrics = _evaluate(agent, eval_env, eval_dataset, config, greedy=True)
            eval_elapsed = time.time() - eval_start

            eval_record = {"episode": episode, "elapsed": eval_elapsed, **eval_metrics}
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(eval_record) + "\n")

            print(
                f"  [EVAL] R={eval_metrics['reward_mean']:.4f}+/-{eval_metrics['reward_std']:.4f} "
                f"SP={eval_metrics['sp_mean']:.4f} TC={eval_metrics['tc_mean']:.4f} "
                f"({eval_metrics['n_pairs']} pairs, {eval_elapsed:.1f}s)"
            )

    # Final checkpoint
    save_checkpoint(
        policy, agent.optimizer, episode,
        config.checkpoint_dir,
        filename="checkpoint_final.pt",
        extra={"baseline": agent._baseline},
    )
    print(f"\nTraining complete. {episode} episodes.")
