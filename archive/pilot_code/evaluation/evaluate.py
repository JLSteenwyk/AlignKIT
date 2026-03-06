"""Greedy evaluation of trained agent on held-out data."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from agent.policy_network import PolicyNetwork
from agent.reinforce import ReinforceAgent
from config import Config
from data.dataset import PairwiseDataset, build_datasets
from env.action_space import ActionSpace
from env.alignment_env import AlignmentBanditEnv
from training.checkpointer import load_checkpoint


def evaluate_agent(config: Config, checkpoint_file: str = "checkpoint_latest.pt", max_pairs: int = 500) -> Dict:
    """Run full greedy evaluation on held-out data.

    Args:
        config: Configuration.
        checkpoint_file: Which checkpoint to load.

    Returns:
        Dict with overall and per-set metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, eval_dataset, _ = build_datasets(
        config.data_dir, config.train_ref_sets, config.eval_ref_sets, config.max_seq_length
    )
    if len(eval_dataset) == 0:
        print("No evaluation data available.")
        return {}

    # Build env
    action_space = ActionSpace(config)
    env = AlignmentBanditEnv(eval_dataset, action_space, config)

    # Load agent
    policy = PolicyNetwork(
        input_dim=config.input_dim,
        hidden_sizes=config.hidden_sizes,
        output_dim=config.num_actions,
    )
    ckpt = load_checkpoint(policy, None, config.checkpoint_dir, checkpoint_file, device)
    episode = ckpt.get("episode", 0)
    policy.to(device)
    policy.eval()

    agent = ReinforceAgent(policy, config, device)
    print(f"Loaded checkpoint from episode {episode}")

    # Evaluate (subsample if dataset is large)
    rewards, sps, tcs = [], [], []
    per_set = {}
    action_counts = Counter()

    eval_pairs = list(eval_dataset)
    if len(eval_pairs) > max_pairs:
        random.seed(config.seed)
        eval_pairs = random.sample(eval_pairs, max_pairs)
    print(f"Evaluating on {len(eval_pairs)} pairs...")

    for ref in eval_pairs:
        state = env.reset_with_ref(ref)
        action = agent.select_action_greedy(state)
        reward, info = env.step(action)

        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))

        gap_open, gap_extend, matrix = action_space.decode(action)
        action_counts[action] += 1

        rs = info.get("ref_set", "unknown")
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": [], "actions": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info.get("sp", 0.0))
        per_set[rs]["tcs"].append(info.get("tc", 0.0))
        per_set[rs]["actions"].append(action)

    # Summary
    results = {
        "overall": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "sp_mean": float(np.mean(sps)),
            "sp_std": float(np.std(sps)),
            "tc_mean": float(np.mean(tcs)),
            "tc_std": float(np.std(tcs)),
            "n_pairs": len(rewards),
        },
        "per_set": {},
        "action_distribution": {},
    }

    for rs, data in per_set.items():
        results["per_set"][rs] = {
            "reward_mean": float(np.mean(data["rewards"])),
            "reward_std": float(np.std(data["rewards"])),
            "sp_mean": float(np.mean(data["sps"])),
            "tc_mean": float(np.mean(data["tcs"])),
            "n_pairs": len(data["rewards"]),
        }

        # Action distribution for this set
        set_action_counts = Counter(data["actions"])
        top_actions = set_action_counts.most_common(5)
        results["per_set"][rs]["top_actions"] = [
            {
                "action": a,
                "count": c,
                "params": dict(zip(
                    ["gap_open", "gap_extend", "matrix"],
                    action_space.decode(a),
                )),
            }
            for a, c in top_actions
        ]

    # Overall action distribution
    top_overall = action_counts.most_common(10)
    results["action_distribution"]["top_10"] = [
        {
            "action": a,
            "count": c,
            "params": dict(zip(
                ["gap_open", "gap_extend", "matrix"],
                action_space.decode(a),
            )),
        }
        for a, c in top_overall
    ]

    # Matrix distribution
    matrix_counts = Counter()
    for a, c in action_counts.items():
        _, _, mat = action_space.decode(a)
        matrix_counts[mat] += c
    results["action_distribution"]["matrices"] = dict(matrix_counts)

    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Overall: R={results['overall']['reward_mean']:.4f} +/- {results['overall']['reward_std']:.4f}")
    print(f"         SP={results['overall']['sp_mean']:.4f}  TC={results['overall']['tc_mean']:.4f}")
    print(f"         N={results['overall']['n_pairs']} pairs")
    for rs, data in sorted(results["per_set"].items()):
        print(f"  {rs}: R={data['reward_mean']:.4f} SP={data['sp_mean']:.4f} TC={data['tc_mean']:.4f} (n={data['n_pairs']})")
    print(f"\nMatrix distribution: {dict(matrix_counts)}")

    # Save results
    results_path = config.log_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results
