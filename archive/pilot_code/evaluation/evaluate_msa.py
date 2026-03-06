"""Greedy evaluation of trained MSA agent on held-out data."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from config_msa import MSAConfig
from data.msa_dataset import build_msa_datasets
from env.msa_action_space import MSAActionSpace
from env.msa_alignment_env import MSAAlignmentEnv
from training.checkpointer import load_checkpoint


def evaluate_msa_agent(
    config: MSAConfig,
    checkpoint_file: str = "checkpoint_latest.pt",
    max_cases: int = None,
) -> Dict:
    """Run full greedy evaluation on held-out MSA data.

    Args:
        config: MSA configuration.
        checkpoint_file: Which checkpoint to load.
        max_cases: Max test cases to evaluate (None = all).

    Returns:
        Dict with overall and per-set metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchical = config.policy_type == "hierarchical"

    # Load data
    benchmark_dir = config.benchmark_dir if config.use_benchmarks else None
    _, eval_dataset = build_msa_datasets(
        config.data_dir, config.all_ref_sets, config.split_ratio, config.split_seed,
        benchmark_dir=benchmark_dir,
    )
    if len(eval_dataset) == 0:
        print("No evaluation data available.")
        return {}

    # Build env
    action_space = MSAActionSpace(config)
    env = MSAAlignmentEnv(eval_dataset, action_space, config)

    # Build policy + agent based on policy_type
    if hierarchical:
        from agent.hierarchical_policy import HierarchicalPolicy
        from agent.reinforce_hierarchical import HierarchicalReinforceAgent

        policy = HierarchicalPolicy(
            input_dim=config.input_dim,
            hidden_sizes=config.hidden_sizes,
            param_sizes=action_space.param_sizes,
        )
        ckpt = load_checkpoint(policy, None, config.checkpoint_dir, checkpoint_file, device)
        policy.to(device)
        policy.eval()
        agent = HierarchicalReinforceAgent(policy, config, device)
    else:
        from agent.policy_network import PolicyNetwork
        from agent.reinforce import ReinforceAgent

        policy = PolicyNetwork(
            input_dim=config.input_dim,
            hidden_sizes=config.hidden_sizes,
            output_dim=config.num_actions,
        )
        ckpt = load_checkpoint(policy, None, config.checkpoint_dir, checkpoint_file, device)
        policy.to(device)
        policy.eval()
        agent = ReinforceAgent(policy, config, device)

    episode = ckpt.get("episode", 0)
    print(f"Loaded checkpoint from episode {episode} (policy_type={config.policy_type})")

    # Evaluate
    rewards, sps, tcs = [], [], []
    per_set = {}
    action_counts = Counter()
    tool_counts = Counter()
    elapsed_times = []

    eval_cases = list(eval_dataset)
    if max_cases and len(eval_cases) > max_cases:
        random.seed(config.seed)
        eval_cases = random.sample(eval_cases, max_cases)
    print(f"Evaluating on {len(eval_cases)} MSA test cases...")

    for case in eval_cases:
        state = env.reset_with_case(case)

        if hierarchical:
            tool_idx, param_idx = agent.select_action_greedy(state)
            reward, info = env.step_hierarchical(tool_idx, param_idx)
            action_key = (tool_idx, param_idx)
        else:
            action = agent.select_action_greedy(state)
            reward, info = env.step(action)
            action_key = action

        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))
        elapsed_times.append(info.get("elapsed", 0.0))

        action_counts[action_key] += 1
        tool_counts[info.get("tool", "unknown")] += 1

        rs = info.get("ref_set", "unknown")
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": [], "actions": [], "tools": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info.get("sp", 0.0))
        per_set[rs]["tcs"].append(info.get("tc", 0.0))
        per_set[rs]["actions"].append(action_key)
        per_set[rs]["tools"].append(info.get("tool", "unknown"))

    # Summary
    results = {
        "overall": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "sp_mean": float(np.mean(sps)),
            "sp_std": float(np.std(sps)),
            "tc_mean": float(np.mean(tcs)),
            "tc_std": float(np.std(tcs)),
            "avg_elapsed": float(np.mean(elapsed_times)),
            "n_cases": len(rewards),
        },
        "tool_distribution": dict(tool_counts),
        "per_set": {},
        "action_distribution": {},
    }

    for rs, data in per_set.items():
        rs_tool_counts = Counter(data["tools"])
        results["per_set"][rs] = {
            "reward_mean": float(np.mean(data["rewards"])),
            "reward_std": float(np.std(data["rewards"])),
            "sp_mean": float(np.mean(data["sps"])),
            "tc_mean": float(np.mean(data["tcs"])),
            "n_cases": len(data["rewards"]),
            "tool_distribution": dict(rs_tool_counts),
        }

        # Top actions for this set
        set_action_counts = Counter(data["actions"])
        top_actions = set_action_counts.most_common(5)
        if hierarchical:
            results["per_set"][rs]["top_actions"] = [
                {"action": str(a), "count": c, "desc": action_space.describe_hierarchical(a[0], a[1])}
                for a, c in top_actions
            ]
        else:
            results["per_set"][rs]["top_actions"] = [
                {"action": a, "count": c, "desc": action_space.describe(a)}
                for a, c in top_actions
            ]

    # Overall action distribution
    top_overall = action_counts.most_common(10)
    if hierarchical:
        results["action_distribution"]["top_10"] = [
            {"action": str(a), "count": c, "desc": action_space.describe_hierarchical(a[0], a[1])}
            for a, c in top_overall
        ]
    else:
        results["action_distribution"]["top_10"] = [
            {"action": a, "count": c, "desc": action_space.describe(a)}
            for a, c in top_overall
        ]

    # Print summary
    print("\n=== MSA Evaluation Results ===")
    print(f"Overall: R={results['overall']['reward_mean']:.4f} +/- {results['overall']['reward_std']:.4f}")
    print(f"         SP={results['overall']['sp_mean']:.4f}  TC={results['overall']['tc_mean']:.4f}")
    print(f"         N={results['overall']['n_cases']} MSAs, avg_time={results['overall']['avg_elapsed']:.1f}s")
    print(f"Tool distribution: {dict(tool_counts)}")
    for rs, data in sorted(results["per_set"].items()):
        print(f"  {rs}: R={data['reward_mean']:.4f} SP={data['sp_mean']:.4f} TC={data['tc_mean']:.4f} (n={data['n_cases']})")

    # Save results
    config.ensure_dirs()
    results_path = config.log_dir / "msa_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results
