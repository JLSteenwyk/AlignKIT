"""Baseline alignment strategies for comparison."""

import random
from typing import Dict, List, Tuple

import numpy as np

from config import Config
from data.balibase_parser import PairwiseReference
from data.dataset import PairwiseDataset
from env.action_space import ActionSpace
from env.alignment_env import AlignmentBanditEnv


def default_baseline(
    dataset: PairwiseDataset, action_space: ActionSpace, config: Config,
    max_pairs: int = 200,
) -> Dict:
    """Evaluate the default BLOSUM62, gap_open=-10, gap_extend=-0.5 baseline.

    Finds the closest action in the discrete space to these default parameters.
    """
    # Find action closest to defaults
    best_action = _find_closest_action(action_space, -10.0, -0.5, "BLOSUM62")

    env = AlignmentBanditEnv(dataset, action_space, config)
    return _evaluate_fixed_action(env, dataset, best_action, action_space, max_pairs)


def grid_search_oracle(
    dataset: PairwiseDataset, action_space: ActionSpace, config: Config,
    max_pairs: int = 50,
) -> Dict:
    """Oracle: try all actions per pair, pick best (upper bound on performance).

    This is slow (500 alignments per pair), so we cap max_pairs.
    """
    env = AlignmentBanditEnv(dataset, action_space, config)
    rewards, sps, tcs = [], [], []

    pairs = list(dataset)
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    for ref in pairs:
        best_reward = -1.0
        best_sp, best_tc = 0.0, 0.0

        for action_idx in range(len(action_space)):
            env.reset_with_ref(ref)
            reward, info = env.step(action_idx)
            if reward > best_reward:
                best_reward = reward
                best_sp = info.get("sp", 0.0)
                best_tc = info.get("tc", 0.0)

        rewards.append(best_reward)
        sps.append(best_sp)
        tcs.append(best_tc)

    return {
        "name": "grid_search_oracle",
        "reward_mean": float(np.mean(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "reward_std": float(np.std(rewards)),
        "n_pairs": len(rewards),
    }


def per_set_best_fixed(
    dataset: PairwiseDataset, action_space: ActionSpace, config: Config,
    max_pairs_per_action: int = 50,
) -> Dict:
    """Find the best single fixed action per reference set.

    For each reference set, evaluates all actions and returns the best one.
    """
    by_set = dataset.get_by_ref_set()
    results = {}

    for ref_set, refs in by_set.items():
        env = AlignmentBanditEnv(
            PairwiseDataset(refs, config.max_seq_length), action_space, config
        )
        sample_refs = refs[:max_pairs_per_action] if len(refs) > max_pairs_per_action else refs

        best_action = 0
        best_mean_reward = -1.0

        # Try each action
        for action_idx in range(len(action_space)):
            action_rewards = []
            for ref in sample_refs:
                env.reset_with_ref(ref)
                reward, _ = env.step(action_idx)
                action_rewards.append(reward)

            mean_r = np.mean(action_rewards)
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                best_action = action_idx

        # Evaluate best action on full set
        gap_open, gap_extend, matrix = action_space.decode(best_action)
        full_rewards = []
        for ref in refs:
            env.reset_with_ref(ref)
            reward, _ = env.step(best_action)
            full_rewards.append(reward)

        results[ref_set] = {
            "best_action": best_action,
            "gap_open": gap_open,
            "gap_extend": gap_extend,
            "matrix": matrix,
            "reward_mean": float(np.mean(full_rewards)),
            "reward_std": float(np.std(full_rewards)),
            "n_pairs": len(full_rewards),
        }

    return {"name": "per_set_best_fixed", "per_set": results}


def _find_closest_action(
    action_space: ActionSpace, target_go: float, target_ge: float, target_matrix: str
) -> int:
    """Find the action index closest to target parameters."""
    best_action = 0
    best_dist = float("inf")

    for idx in range(len(action_space)):
        go, ge, mat = action_space.decode(idx)
        if mat != target_matrix:
            continue
        dist = (go - target_go) ** 2 + (ge - target_ge) ** 2
        if dist < best_dist:
            best_dist = dist
            best_action = idx

    return best_action


def _evaluate_fixed_action(
    env: AlignmentBanditEnv, dataset: PairwiseDataset, action_idx: int,
    action_space: ActionSpace, max_pairs: int,
) -> Dict:
    """Evaluate a single fixed action on the dataset."""
    rewards, sps, tcs = [], [], []
    per_set = {}

    pairs = list(dataset)
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    gap_open, gap_extend, matrix = action_space.decode(action_idx)

    for ref in pairs:
        env.reset_with_ref(ref)
        reward, info = env.step(action_idx)
        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))

        rs = info.get("ref_set", "unknown")
        per_set.setdefault(rs, []).append(reward)

    result = {
        "name": "default_blosum62",
        "action_idx": action_idx,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "matrix": matrix,
        "reward_mean": float(np.mean(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "reward_std": float(np.std(rewards)),
        "n_pairs": len(rewards),
    }

    for rs, rs_rewards in per_set.items():
        result[f"{rs}_reward"] = float(np.mean(rs_rewards))

    return result
