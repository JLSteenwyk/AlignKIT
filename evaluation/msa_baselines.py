"""Baseline MSA strategies: MAFFT default, MUSCLE default, and Clustal Omega default."""

from collections import Counter
from typing import Dict

import numpy as np

from config_msa import MSAConfig
from data.msa_dataset import MSADataset
from env.msa_action_space import MSAActionSpace
from env.msa_alignment_env import MSAAlignmentEnv


def _find_mafft_default_action(action_space: MSAActionSpace) -> int:
    """Find MAFFT default action: auto strategy, op=1.53, ep=0.0, maxiter=0."""
    for idx in range(action_space.n_mafft):
        action = action_space.decode(idx)
        if (
            action.mafft_strategy == "auto"
            and action.mafft_op == 1.53
            and action.mafft_ep == 0.0
            and action.mafft_maxiterate == 0
        ):
            return idx
    # Fallback: first MAFFT action
    return 0


def _find_muscle_default_action(action_space: MSAActionSpace) -> int:
    """Find MUSCLE default action: -align, perm=none, perturb=0."""
    for idx in range(action_space.n_mafft, action_space.n_actions):
        action = action_space.decode(idx)
        if (
            action.muscle_command == "align"
            and action.muscle_perm == "none"
            and action.muscle_perturb == 0
        ):
            return idx
    # Fallback: first MUSCLE action
    return action_space.n_mafft


def _evaluate_fixed_msa_action(
    env: MSAAlignmentEnv, dataset: MSADataset, action_idx: int,
    action_space: MSAActionSpace,
) -> Dict:
    """Evaluate a single fixed action on all MSA test cases."""
    rewards, sps, tcs = [], [], []
    per_set = {}
    elapsed_times = []

    for case in dataset:
        env.reset_with_case(case)
        reward, info = env.step(action_idx)

        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))
        elapsed_times.append(info.get("elapsed", 0.0))

        rs = info.get("ref_set", "unknown")
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info.get("sp", 0.0))
        per_set[rs]["tcs"].append(info.get("tc", 0.0))

    action = action_space.decode(action_idx)
    result = {
        "action_idx": action_idx,
        "action_desc": action_space.describe(action_idx),
        "tool": action.tool,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "sp_mean": float(np.mean(sps)) if sps else 0.0,
        "tc_mean": float(np.mean(tcs)) if tcs else 0.0,
        "avg_elapsed": float(np.mean(elapsed_times)) if elapsed_times else 0.0,
        "n_cases": len(rewards),
        "per_set": {},
    }

    for rs, data in per_set.items():
        result["per_set"][rs] = {
            "reward_mean": float(np.mean(data["rewards"])),
            "sp_mean": float(np.mean(data["sps"])),
            "tc_mean": float(np.mean(data["tcs"])),
            "n_cases": len(data["rewards"]),
        }

    return result


def mafft_default_baseline(
    dataset: MSADataset, action_space: MSAActionSpace, config: MSAConfig,
) -> Dict:
    """Evaluate MAFFT with default parameters on all eval MSAs."""
    env = MSAAlignmentEnv(dataset, action_space, config)
    action_idx = _find_mafft_default_action(action_space)
    result = _evaluate_fixed_msa_action(env, dataset, action_idx, action_space)
    result["name"] = "mafft_default"
    return result


def muscle_default_baseline(
    dataset: MSADataset, action_space: MSAActionSpace, config: MSAConfig,
) -> Dict:
    """Evaluate MUSCLE with default parameters on all eval MSAs."""
    env = MSAAlignmentEnv(dataset, action_space, config)
    action_idx = _find_muscle_default_action(action_space)
    result = _evaluate_fixed_msa_action(env, dataset, action_idx, action_space)
    result["name"] = "muscle_default"
    return result


def _find_clustalo_default_action(action_space: MSAActionSpace) -> int:
    """Find Clustal Omega default action: iter=0, full=no, full-iter=no, kimura=no."""
    start = action_space.n_mafft + action_space.n_muscle
    for idx in range(start, action_space.n_actions):
        action = action_space.decode(idx)
        if (
            action.clustalo_iter == 0
            and not action.clustalo_full
            and not action.clustalo_full_iter
            and not action.clustalo_kimura
        ):
            return idx
    # Fallback: first Clustal Omega action
    return start


def clustalo_default_baseline(
    dataset: MSADataset, action_space: MSAActionSpace, config: MSAConfig,
) -> Dict:
    """Evaluate Clustal Omega with default parameters on all eval MSAs."""
    env = MSAAlignmentEnv(dataset, action_space, config)
    action_idx = _find_clustalo_default_action(action_space)
    result = _evaluate_fixed_msa_action(env, dataset, action_idx, action_space)
    result["name"] = "clustalo_default"
    return result
