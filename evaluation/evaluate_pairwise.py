"""Evaluation and baselines for pairwise RL gap penalty learning."""

import time
from collections import defaultdict

import numpy as np
import torch

from agent.continuous_policy import ContinuousPolicy
from agent.reinforce_continuous import ReinforceContinuousAgent
from config import Config
from data.pairwise_dataset import build_pairwise_datasets
from env.needleman_wunsch import needleman_wunsch, encode_sequence
from env.pairwise_env import PairwiseEnv
from scoring.sp_score import sp_score
from scoring.tc_score import tc_score
from scoring.reward import compute_reward
from training.checkpointer import load_checkpoint


# ---- Baselines ----

def nw_default_baseline(eval_dataset, config):
    """NW with standard textbook parameters: BLOSUM62, go=10, ge=0.5.

    Returns dict of metrics.
    """
    K = config.num_regions
    gap_open = np.full(K, 10.0)
    gap_extend = np.full(K, 0.5)

    rewards, sps, tcs = [], [], []
    per_set = defaultdict(lambda: {"rewards": [], "sps": [], "tcs": []})

    for ex in eval_dataset:
        pred1, pred2, _ = needleman_wunsch(ex.seq1_idx, ex.seq2_idx, gap_open, gap_extend, K)
        sp = sp_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        tc = tc_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        r = compute_reward(sp, tc, config.sp_weight, config.tc_weight)
        rewards.append(r)
        sps.append(sp)
        tcs.append(tc)
        per_set[ex.ref_set]["rewards"].append(r)
        per_set[ex.ref_set]["sps"].append(sp)
        per_set[ex.ref_set]["tcs"].append(tc)

    result = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_pairs": len(rewards),
        "gap_open": 10.0,
        "gap_extend": 0.5,
        "per_set": {},
    }
    for rs, data in per_set.items():
        result["per_set"][rs] = {
            "reward_mean": float(np.mean(data["rewards"])),
            "sp_mean": float(np.mean(data["sps"])),
            "tc_mean": float(np.mean(data["tcs"])),
            "n_pairs": len(data["rewards"]),
        }
    return result


def best_constant_baseline(eval_dataset, config, max_pairs=None):
    """Grid search over constant (go, ge) to find best fixed parameters.

    Returns dict of metrics for the best configuration.
    """
    K = config.num_regions

    # Grid of parameters to try
    go_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    ge_values = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    examples = list(eval_dataset)
    if max_pairs and len(examples) > max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(examples), max_pairs, replace=False)
        examples = [examples[i] for i in indices]

    best_reward = -1
    best_go, best_ge = 10.0, 0.5

    for go in go_values:
        for ge in ge_values:
            gap_open = np.full(K, float(go))
            gap_extend = np.full(K, float(ge))
            rewards = []
            for ex in examples:
                pred1, pred2, _ = needleman_wunsch(
                    ex.seq1_idx, ex.seq2_idx, gap_open, gap_extend, K
                )
                sp = sp_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
                tc = tc_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
                rewards.append(compute_reward(sp, tc, config.sp_weight, config.tc_weight))
            mean_r = float(np.mean(rewards))
            if mean_r > best_reward:
                best_reward = mean_r
                best_go, best_ge = go, ge

    # Re-evaluate best on full dataset
    gap_open = np.full(K, float(best_go))
    gap_extend = np.full(K, float(best_ge))
    rewards, sps, tcs = [], [], []
    for ex in eval_dataset:
        pred1, pred2, _ = needleman_wunsch(ex.seq1_idx, ex.seq2_idx, gap_open, gap_extend, K)
        sp = sp_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        tc = tc_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
        rewards.append(compute_reward(sp, tc, config.sp_weight, config.tc_weight))
        sps.append(sp)
        tcs.append(tc)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_pairs": len(rewards),
        "gap_open": best_go,
        "gap_extend": best_ge,
    }


def per_pair_oracle(eval_dataset, config, max_pairs=None):
    """Per-pair oracle: best (go, ge) per pair from grid.

    This is the upper bound on what constant-per-pair NW can achieve.
    """
    K = config.num_regions
    go_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    ge_values = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    examples = list(eval_dataset)
    if max_pairs and len(examples) > max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(examples), max_pairs, replace=False)
        examples = [examples[i] for i in indices]

    best_rewards, best_sps, best_tcs = [], [], []

    for ex in examples:
        best_r = -1
        best_sp, best_tc = 0, 0
        for go in go_values:
            for ge in ge_values:
                gap_open = np.full(K, float(go))
                gap_extend = np.full(K, float(ge))
                pred1, pred2, _ = needleman_wunsch(
                    ex.seq1_idx, ex.seq2_idx, gap_open, gap_extend, K
                )
                sp = sp_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
                tc = tc_score(pred1, pred2, ex.ref_seq1_aligned, ex.ref_seq2_aligned)
                r = compute_reward(sp, tc, config.sp_weight, config.tc_weight)
                if r > best_r:
                    best_r = r
                    best_sp, best_tc = sp, tc
        best_rewards.append(best_r)
        best_sps.append(best_sp)
        best_tcs.append(best_tc)

    return {
        "reward_mean": float(np.mean(best_rewards)),
        "reward_std": float(np.std(best_rewards)),
        "sp_mean": float(np.mean(best_sps)),
        "tc_mean": float(np.mean(best_tcs)),
        "n_pairs": len(best_rewards),
    }


# ---- Agent evaluation ----

def evaluate_pairwise_agent(config: Config, checkpoint: str, max_pairs: int = None):
    """Full evaluation: load agent, run on eval set, compare to baselines."""

    device = torch.device("cpu")

    # Data
    _, eval_dataset = build_pairwise_datasets(
        config.data_dir, config.train_ref_sets, config.eval_ref_sets,
        config.max_seq_length,
    )

    if len(eval_dataset) == 0:
        print("No evaluation data. Run setup.sh to download BAliBASE.")
        return

    eval_examples = list(eval_dataset)
    if max_pairs and len(eval_examples) > max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(eval_examples), max_pairs, replace=False)
        eval_examples = [eval_examples[i] for i in indices]

    # Load agent
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
    load_checkpoint(policy, None, config.checkpoint_dir, checkpoint, device)
    policy.eval()

    env = PairwiseEnv(eval_dataset, config)
    agent = ReinforceContinuousAgent(policy, config, device)

    # Evaluate agent
    print(f"\n=== RL Agent (greedy) on {len(eval_examples)} pairs ===")
    rewards, sps, tcs = [], [], []
    per_set = defaultdict(lambda: {"rewards": [], "sps": [], "tcs": []})

    t0 = time.time()
    for ex in eval_examples:
        state = env.reset_with(ex)
        action = agent.select_action_greedy(state)
        reward, info = env.step(action)
        rewards.append(reward)
        sps.append(info["sp"])
        tcs.append(info["tc"])
        per_set[ex.ref_set]["rewards"].append(reward)
        per_set[ex.ref_set]["sps"].append(info["sp"])
        per_set[ex.ref_set]["tcs"].append(info["tc"])

    elapsed = time.time() - t0
    print(f"  Reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  SP: {np.mean(sps):.4f}  TC: {np.mean(tcs):.4f}")
    print(f"  Time: {elapsed:.1f}s")
    for rs in sorted(per_set.keys()):
        d = per_set[rs]
        print(f"    {rs}: R={np.mean(d['rewards']):.4f} SP={np.mean(d['sps']):.4f} TC={np.mean(d['tcs']):.4f} (n={len(d['rewards'])})")

    # NW default baseline
    print(f"\n=== NW Default (go=10, ge=0.5) ===")
    from data.pairwise_dataset import PairwiseDataset
    eval_ds_subset = PairwiseDataset(eval_examples)
    default_result = nw_default_baseline(eval_ds_subset, config)
    print(f"  Reward: {default_result['reward_mean']:.4f} +/- {default_result['reward_std']:.4f}")
    print(f"  SP: {default_result['sp_mean']:.4f}  TC: {default_result['tc_mean']:.4f}")

    # Best constant baseline
    print(f"\n=== Best Constant NW (grid search) ===")
    best_const = best_constant_baseline(eval_ds_subset, config, max_pairs=min(100, len(eval_examples)))
    print(f"  Reward: {best_const['reward_mean']:.4f} +/- {best_const['reward_std']:.4f}")
    print(f"  SP: {best_const['sp_mean']:.4f}  TC: {best_const['tc_mean']:.4f}")
    print(f"  Best: go={best_const['gap_open']}, ge={best_const['gap_extend']}")

    # Per-pair oracle
    print(f"\n=== Per-Pair Oracle ===")
    oracle = per_pair_oracle(eval_ds_subset, config, max_pairs=min(50, len(eval_examples)))
    print(f"  Reward: {oracle['reward_mean']:.4f} +/- {oracle['reward_std']:.4f}")
    print(f"  SP: {oracle['sp_mean']:.4f}  TC: {oracle['tc_mean']:.4f}")
