"""Training loop for MSA RL parameter learning (continuous op/ep for MAFFT)."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from agent.continuous_policy import ContinuousPolicy
from agent.reinforce_continuous import ReinforceContinuousAgent
from config import MSAConfig
from data.msa_dataset import MSADataset, load_msa_test_cases
from env.msa_env import MSAEnv
from training.checkpointer import save_checkpoint, load_checkpoint


def _evaluate(agent, eval_env, eval_dataset, greedy=True):
    """Run evaluation on the eval dataset. Returns dict of metrics."""
    rewards, sps, tcs = [], [], []
    per_set = {}

    for case in eval_dataset:
        state = eval_env.reset_with(case)
        if greedy:
            action = agent.select_action_greedy(state)
        else:
            action, _, _, _ = agent.select_action(state)
        reward, info = eval_env.step(action)
        rewards.append(reward)
        sps.append(info["sp"])
        tcs.append(info["tc"])

        rs = case.ref_set
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info["sp"])
        per_set[rs]["tcs"].append(info["tc"])

    result = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_cases": len(rewards),
        "per_set": {},
    }
    for rs, data in sorted(per_set.items()):
        result["per_set"][rs] = {
            "reward_mean": float(np.mean(data["rewards"])),
            "sp_mean": float(np.mean(data["sps"])),
            "tc_mean": float(np.mean(data["tcs"])),
            "n_cases": len(data["rewards"]),
        }
    return result


def train_msa(config: MSAConfig, resume_from: str = None):
    """Main training loop for MSA alignment RL."""

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cpu")

    # Data — split by ref_set
    print("Loading training MSA test cases...")
    train_cases = load_msa_test_cases(config.data_dir, config.train_ref_sets)
    print("Loading evaluation MSA test cases...")
    eval_cases = load_msa_test_cases(config.data_dir, config.eval_ref_sets)

    train_dataset = MSADataset(train_cases)
    eval_dataset = MSADataset(eval_cases)

    if len(train_dataset) == 0:
        print("No training data found. Check BAliBASE data path.")
        return

    # Environment
    train_env = MSAEnv(train_dataset, config)
    eval_env = MSAEnv(eval_dataset, config)

    # Policy + Agent
    # For MSA: action_dim=2 (op, ep), treat as 1 "region" with op as gap_open, ep as gap_extend
    policy = ContinuousPolicy(
        input_dim=config.input_dim,
        hidden_sizes=config.hidden_sizes,
        action_dim=config.action_dim,
        initial_log_std=config.initial_log_std,
        min_std=config.min_std,
        gap_open_range=(config.op_min, config.op_max),
        gap_extend_range=(config.ep_min, config.ep_max),
        num_regions=1,
    )
    agent = ReinforceContinuousAgent(policy, config, device)

    start_episode = 0

    if resume_from:
        ckpt = load_checkpoint(policy, agent.optimizer, config.checkpoint_dir, resume_from, device)
        start_episode = ckpt.get("episode", 0)
        print("Resumed from episode %d" % start_episode)

    # Logging
    config.log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = config.log_dir / "msa_train.jsonl"
    eval_log_path = config.log_dir / "msa_eval.jsonl"

    print("\nStarting MSA training: %d episodes, batch_size=%d" % (config.num_episodes, config.batch_size))
    print("Train: %d MSAs, Eval: %d MSAs" % (len(train_dataset), len(eval_dataset)))
    print("Action: op in [%.1f, %.1f], ep in [%.1f, %.1f]\n" % (
        config.op_min, config.op_max, config.ep_min, config.ep_max))

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
            "op_mean": float(np.mean([i["op"] for i in batch_infos])),
            "ep_mean": float(np.mean([i["ep"] for i in batch_infos])),
            "elapsed": batch_elapsed,
            **metrics,
        }
        with open(train_log_path, "a") as f:
            f.write(json.dumps(train_record) + "\n")

        # Print progress
        if batch_count % 10 == 0:
            avg_op = np.mean([i["op"] for i in batch_infos])
            avg_ep = np.mean([i["ep"] for i in batch_infos])
            errors = sum(1 for i in batch_infos if "error" in i)
            err_str = " err=%d" % errors if errors else ""
            print(
                "[Ep %6d] R=%.4f SP=%.4f TC=%.4f "
                "loss=%.4f ent=%.2f "
                "op=%.2f ep=%.3f%s "
                "(%.1fs)" % (
                    episode, train_record["reward_mean"],
                    train_record["sp_mean"], train_record["tc_mean"],
                    metrics["loss"], metrics["entropy"],
                    avg_op, avg_ep, err_str,
                    batch_elapsed,
                )
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
            eval_metrics = _evaluate(agent, eval_env, eval_dataset, greedy=True)
            eval_elapsed = time.time() - eval_start

            eval_record = {"episode": episode, "elapsed": eval_elapsed, **eval_metrics}
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(eval_record) + "\n")

            print(
                "  [EVAL] R=%.4f+/-%.4f SP=%.4f TC=%.4f (%d cases, %.1fs)" % (
                    eval_metrics["reward_mean"], eval_metrics["reward_std"],
                    eval_metrics["sp_mean"], eval_metrics["tc_mean"],
                    eval_metrics["n_cases"], eval_elapsed,
                )
            )
            for rs, data in sorted(eval_metrics["per_set"].items()):
                print(
                    "    %s: R=%.4f SP=%.4f TC=%.4f (n=%d)" % (
                        rs, data["reward_mean"], data["sp_mean"],
                        data["tc_mean"], data["n_cases"],
                    )
                )

    # Final checkpoint
    save_checkpoint(
        policy, agent.optimizer, episode,
        config.checkpoint_dir,
        filename="checkpoint_final.pt",
        extra={"baseline": agent._baseline},
    )
    print("\nMSA training complete. %d episodes." % episode)
