"""MAFFT-only oracle: evaluate all 144 MAFFT configs on full eval set."""

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_msa import MSAConfig
from data.msa_dataset import build_msa_datasets
from env.msa_action_space import MSAActionSpace
from env.msa_alignment_env import MSAAlignmentEnv


def main():
    config = MSAConfig()
    config.subprocess_timeout = 120

    train_dataset, eval_dataset = build_msa_datasets(
        config.data_dir, config.all_ref_sets, config.split_ratio, config.split_seed
    )

    action_space = MSAActionSpace(config)
    env = MSAAlignmentEnv(eval_dataset, action_space, config)

    # MAFFT actions are 0..n_mafft-1
    n_mafft = action_space.n_mafft
    eval_cases = list(eval_dataset)

    print("=" * 70)
    print("MAFFT-ONLY ORACLE (full eval set)")
    print("=" * 70)
    print(f"MAFFT actions: {n_mafft}")
    print(f"Eval cases: {len(eval_cases)}")
    print(f"Total alignments: {n_mafft * len(eval_cases)}")
    print()
    sys.stdout.flush()

    rewards = np.full((len(eval_cases), n_mafft), np.nan)
    sp_scores = np.full((len(eval_cases), n_mafft), np.nan)
    tc_scores = np.full((len(eval_cases), n_mafft), np.nan)
    errors = np.zeros((len(eval_cases), n_mafft), dtype=bool)

    case_ids = []
    case_ref_sets = []
    total_start = time.time()

    for i, case in enumerate(eval_cases):
        case_ids.append(case.case_id)
        case_ref_sets.append(case.ref_set)
        case_start = time.time()

        for a in range(n_mafft):
            env.reset_with_case(case)
            reward, info = env.step(a)
            rewards[i, a] = reward
            sp_scores[i, a] = info.get("sp", 0.0)
            tc_scores[i, a] = info.get("tc", 0.0)
            if "error" in info:
                errors[i, a] = True

        case_elapsed = time.time() - case_start
        total_elapsed = time.time() - total_start
        best_r = np.nanmax(rewards[i])
        best_a = int(np.nanargmax(rewards[i]))
        avg_r = np.nanmean(rewards[i])
        n_err = errors[i].sum()
        eta = (total_elapsed / (i + 1)) * (len(eval_cases) - i - 1)

        print(
            f"  Case {i+1:3d}/{len(eval_cases)} [{case.ref_set:<8s}/{case.case_id:<20s}]: "
            f"best R={best_r:.4f} (a={best_a}) avg={avg_r:.4f} err={n_err} | "
            f"{case_elapsed:.0f}s (ETA {eta:.0f}s)"
        )
        sys.stdout.flush()

    # Save raw results
    output_path = os.path.join(os.path.dirname(__file__), "oracle_mafft_only_results.npz")
    np.savez(
        output_path,
        rewards=rewards, sp=sp_scores, tc=tc_scores, errors=errors,
        case_ids=case_ids, case_ref_sets=case_ref_sets,
    )

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    best_per_case_r = np.nanmax(rewards, axis=1)
    best_per_case_idx = np.nanargmax(rewards, axis=1)
    best_per_case_sp = np.array([sp_scores[i, best_per_case_idx[i]] for i in range(len(eval_cases))])
    best_per_case_tc = np.array([tc_scores[i, best_per_case_idx[i]] for i in range(len(eval_cases))])

    oracle_r = best_per_case_r.mean()
    print(f"\n1. MAFFT ORACLE (best config per case, {len(eval_cases)} cases):")
    print(f"   Reward: {oracle_r:.4f} +/- {best_per_case_r.std():.4f}")
    print(f"   SP:     {best_per_case_sp.mean():.4f}")
    print(f"   TC:     {best_per_case_tc.mean():.4f}")

    # Best fixed action
    mean_per_action = np.nanmean(rewards, axis=0)
    top_actions = np.argsort(mean_per_action)[::-1][:10]
    best_fixed_r = mean_per_action[top_actions[0]]

    print(f"\n2. TOP 10 FIXED MAFFT CONFIGS:")
    for rank, a in enumerate(top_actions):
        desc = action_space.describe(int(a))
        print(f"   #{rank+1:2d}  action {a:3d}: R={mean_per_action[a]:.4f}  "
              f"SP={np.nanmean(sp_scores[:, a]):.4f}  TC={np.nanmean(tc_scores[:, a]):.4f}  | {desc}")

    # Per-dataset breakdown
    print(f"\n3. PER-DATASET BREAKDOWN:")
    print(f"   {'Dataset':<12s}  {'Oracle':>7s}  {'BstFix':>7s}  {'Spread':>7s}  {'N':>3s}")
    print("   " + "-" * 50)
    per_set_oracles = {}
    for rs in sorted(set(case_ref_sets)):
        mask = [j for j, r in enumerate(case_ref_sets) if r == rs]
        ds_oracle = best_per_case_r[mask].mean()
        ds_best_fixed = np.nanmean(rewards[mask], axis=0).max()
        spread = ds_oracle - ds_best_fixed
        per_set_oracles[rs] = float(ds_oracle)
        print(f"   {rs:<12s}  {ds_oracle:7.4f}  {ds_best_fixed:7.4f}  {spread:+7.4f}  {len(mask):3d}")

    # Action diversity
    unique_best = len(set(int(x) for x in best_per_case_idx))
    print(f"\n4. ACTION DIVERSITY: {unique_best} unique best actions out of {n_mafft}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save summary
    summary = {
        "n_actions": n_mafft,
        "n_cases": len(eval_cases),
        "oracle_reward": float(oracle_r),
        "oracle_sp": float(best_per_case_sp.mean()),
        "oracle_tc": float(best_per_case_tc.mean()),
        "best_fixed_action": int(top_actions[0]),
        "best_fixed_reward": float(best_fixed_r),
        "unique_best_actions": unique_best,
        "per_set_oracle": per_set_oracles,
        "total_time_s": total_time,
    }
    summary_path = os.path.join(os.path.dirname(__file__), "oracle_mafft_only_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    print(f"Raw results saved to {output_path}")


if __name__ == "__main__":
    main()
