"""Oracle analysis: evaluate every action on eval cases to find performance ceiling.

Computes:
  1. Oracle upper bound (best action per case)
  2. Best fixed action (single best action across all cases)
  3. Best per-tool configs
  4. Per-dataset breakdown
  5. Headroom vs current agent
"""

import json
import os
import sys
import time
import random
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_msa import MSAConfig
from data.msa_dataset import build_msa_datasets
from env.msa_action_space import MSAActionSpace
from env.msa_alignment_env import MSAAlignmentEnv


def stratified_subsample(eval_cases, max_per_dataset=8, seed=42):
    """Stratified subsample: up to max_per_dataset cases per ref_set."""
    by_dataset = defaultdict(list)
    for case in eval_cases:
        by_dataset[case.ref_set].append(case)

    selected = []
    rng = random.Random(seed)
    for rs in sorted(by_dataset.keys()):
        cases = by_dataset[rs]
        if len(cases) > max_per_dataset:
            selected.extend(rng.sample(cases, max_per_dataset))
        else:
            selected.extend(cases)

    return selected, by_dataset


def main():
    config = MSAConfig()
    # BAliBASE-only (default: use_benchmarks=False)
    # Shorter timeout for oracle — skip slow configs rather than blocking
    config.subprocess_timeout = 120

    benchmark_dir = config.benchmark_dir if config.use_benchmarks else None
    train_dataset, eval_dataset = build_msa_datasets(
        config.data_dir, config.all_ref_sets, config.split_ratio, config.split_seed,
        benchmark_dir=benchmark_dir,
        subsample_benchmarks_to=config.subsample_benchmarks_to if config.use_benchmarks else None,
    )

    action_space = MSAActionSpace(config)
    env = MSAAlignmentEnv(eval_dataset, action_space, config)

    n_actions = action_space.n_actions
    eval_cases = list(eval_dataset)
    selected, by_dataset = stratified_subsample(eval_cases, max_per_dataset=8)

    n_mafft = action_space.n_mafft
    n_muscle = action_space.n_muscle
    mafft_end = n_mafft
    muscle_end = n_mafft + n_muscle

    print("=" * 70)
    print("ORACLE ANALYSIS")
    print("=" * 70)
    print("Action space: %d actions (MAFFT: %d, MUSCLE: %d, ClustalO: %d)" % (
        n_actions, n_mafft, n_muscle, action_space.n_clustalo))
    print("Eval cases: %d total, %d selected (stratified subsample)" % (
        len(eval_cases), len(selected)))
    print("Cases per dataset: %s" % ", ".join(
        "%s=%d" % (k, min(len(v), 8)) for k, v in sorted(by_dataset.items())))
    print("Total alignments: %d" % (n_actions * len(selected)))
    print()
    sys.stdout.flush()

    # Results matrices: [n_cases, n_actions]
    rewards = np.full((len(selected), n_actions), np.nan)
    sp_scores = np.full((len(selected), n_actions), np.nan)
    tc_scores = np.full((len(selected), n_actions), np.nan)
    elapsed_times = np.full((len(selected), n_actions), np.nan)
    errors = np.zeros((len(selected), n_actions), dtype=bool)

    case_ids = []
    case_ref_sets = []

    total_start = time.time()

    for i, case in enumerate(selected):
        case_ids.append(case.case_id)
        case_ref_sets.append(case.ref_set)
        case_start = time.time()

        for a in range(n_actions):
            env.reset_with_case(case)
            reward, info = env.step(a)
            rewards[i, a] = reward
            sp_scores[i, a] = info.get("sp", 0.0)
            tc_scores[i, a] = info.get("tc", 0.0)
            elapsed_times[i, a] = info.get("elapsed", 0.0)
            if "error" in info:
                errors[i, a] = True

        case_elapsed = time.time() - case_start
        total_elapsed = time.time() - total_start
        best_r = np.nanmax(rewards[i])
        best_a = np.nanargmax(rewards[i])
        best_tool = action_space.decode(int(best_a)).tool
        avg_r = np.nanmean(rewards[i])
        n_err = errors[i].sum()

        cases_remaining = len(selected) - (i + 1)
        avg_per_case = total_elapsed / (i + 1)
        eta = avg_per_case * cases_remaining

        print("  Case %3d/%d [%-8s/%-20s]: best R=%.4f (%s a=%d) avg=%.4f err=%d | %.0fs (ETA %.0fs)" % (
            i + 1, len(selected), case.ref_set, case.case_id,
            best_r, best_tool, best_a, avg_r, n_err,
            case_elapsed, eta))
        sys.stdout.flush()

    # Save raw results
    output_path = os.path.join(os.path.dirname(__file__), "oracle_results.npz")
    np.savez(output_path,
             rewards=rewards, sp=sp_scores, tc=tc_scores,
             elapsed=elapsed_times, errors=errors,
             case_ids=case_ids, case_ref_sets=case_ref_sets)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # 1. Oracle upper bound: best action per case
    best_per_case_r = np.nanmax(rewards, axis=1)
    best_per_case_idx = np.nanargmax(rewards, axis=1)
    best_per_case_sp = np.array([sp_scores[i, best_per_case_idx[i]] for i in range(len(selected))])
    best_per_case_tc = np.array([tc_scores[i, best_per_case_idx[i]] for i in range(len(selected))])

    print("\n1. ORACLE (best action per case):")
    print("   Reward: %.4f +/- %.4f" % (best_per_case_r.mean(), best_per_case_r.std()))
    print("   SP:     %.4f" % best_per_case_sp.mean())
    print("   TC:     %.4f" % best_per_case_tc.mean())

    # Which tool is best per case?
    oracle_tools = []
    for i in range(len(selected)):
        a = int(best_per_case_idx[i])
        oracle_tools.append(action_space.decode(a).tool)
    tool_dist = Counter(oracle_tools)
    print("   Tool distribution: %s" % dict(tool_dist))

    # 2. Best fixed action (single action for all cases)
    mean_per_action = np.nanmean(rewards, axis=0)
    top_k = 10
    top_actions = np.argsort(mean_per_action)[::-1][:top_k]

    print("\n2. TOP %d FIXED ACTIONS (single action for all cases):" % top_k)
    for rank, a in enumerate(top_actions):
        desc = action_space.describe(int(a))
        tool = action_space.decode(int(a)).tool
        print("   #%2d  action %3d [%-8s]: R=%.4f  SP=%.4f  TC=%.4f  | %s" % (
            rank + 1, a, tool, mean_per_action[a],
            np.nanmean(sp_scores[:, a]), np.nanmean(tc_scores[:, a]),
            desc))

    # 3. Best per tool
    print("\n3. BEST CONFIG PER TOOL:")
    for tool_name, start, end in [("MAFFT", 0, mafft_end),
                                   ("MUSCLE", mafft_end, muscle_end),
                                   ("ClustalO", muscle_end, n_actions)]:
        tool_means = mean_per_action[start:end]
        best_in_tool = tool_means.argmax()
        best_global = start + best_in_tool
        desc = action_space.describe(int(best_global))
        print("   %-8s: action %3d  R=%.4f  SP=%.4f  TC=%.4f" % (
            tool_name, best_global, tool_means[best_in_tool],
            np.nanmean(sp_scores[:, best_global]),
            np.nanmean(tc_scores[:, best_global])))
        print("            %s" % desc)

        # Also show worst for context
        worst_in_tool = tool_means.argmin()
        worst_global = start + worst_in_tool
        print("   %-8s  worst: action %3d  R=%.4f" % (
            "", worst_global, tool_means[worst_in_tool]))

    # 4. Per-dataset breakdown
    print("\n4. PER-DATASET BREAKDOWN:")
    print("   %-12s  %7s  %7s  %7s  %7s  %s" % (
        "Dataset", "Oracle", "BstFix", "AvgAll", "Spread", "OracleTool"))
    print("   " + "-" * 70)
    for rs in sorted(set(case_ref_sets)):
        mask = [j for j, r in enumerate(case_ref_sets) if r == rs]
        oracle_r = best_per_case_r[mask].mean()
        # Best fixed action for this dataset
        ds_mean_per_action = np.nanmean(rewards[mask], axis=0)
        best_fixed_r = ds_mean_per_action.max()
        avg_all_r = ds_mean_per_action.mean()
        spread = oracle_r - best_fixed_r
        # Oracle tool dist for this dataset
        ds_tools = [oracle_tools[j] for j in mask]
        ds_tool_dist = Counter(ds_tools)
        tool_str = " ".join("%s=%d" % (t, c) for t, c in sorted(ds_tool_dist.items()))
        print("   %-12s  %7.4f  %7.4f  %7.4f  %+7.4f  %s" % (
            rs, oracle_r, best_fixed_r, avg_all_r, spread, tool_str))

    # 5. Headroom analysis
    current_agent = 0.560  # From latest eval
    best_fixed = mean_per_action.max()
    oracle = best_per_case_r.mean()

    # Best-per-tool oracle (pick best within each tool, then best tool per case)
    best_tool_per_case = np.zeros(len(selected))
    for i in range(len(selected)):
        mafft_best = np.nanmax(rewards[i, :mafft_end])
        muscle_best = np.nanmax(rewards[i, mafft_end:muscle_end])
        clustalo_best = np.nanmax(rewards[i, muscle_end:])
        best_tool_per_case[i] = max(mafft_best, muscle_best, clustalo_best)

    print("\n5. HEADROOM ANALYSIS:")
    print("   Current agent (greedy):       %.4f" % current_agent)
    print("   Best fixed action:            %.4f  (headroom: %+.4f)" % (best_fixed, best_fixed - current_agent))
    print("   Oracle (best per case):       %.4f  (headroom: %+.4f)" % (oracle, oracle - current_agent))
    print()
    print("   Breakdown of oracle gain:")
    print("   - From better MAFFT params:   agent picks one config, oracle picks best per case")
    mafft_only_oracle = np.nanmax(rewards[:, :mafft_end], axis=1).mean()
    muscle_only_oracle = np.nanmax(rewards[:, mafft_end:muscle_end], axis=1).mean()
    clustalo_only_oracle = np.nanmax(rewards[:, muscle_end:], axis=1).mean()
    print("   - MAFFT-only oracle:          %.4f" % mafft_only_oracle)
    print("   - MUSCLE-only oracle:         %.4f" % muscle_only_oracle)
    print("   - ClustalO-only oracle:       %.4f" % clustalo_only_oracle)
    print("   - Full oracle (any tool):     %.4f" % oracle)

    # 6. Action diversity: how many unique best actions are there?
    unique_best = len(set(int(x) for x in best_per_case_idx))
    print("\n6. ACTION DIVERSITY:")
    print("   Unique best actions across %d cases: %d / %d possible" % (
        len(selected), unique_best, n_actions))
    best_action_counts = Counter(int(x) for x in best_per_case_idx)
    print("   Most common oracle actions:")
    for a, count in best_action_counts.most_common(10):
        desc = action_space.describe(a)
        tool = action_space.decode(a).tool
        print("     action %3d [%-8s] x%d: %s" % (a, tool, count, desc))

    # 7. Per-action variance: which actions are most consistent?
    print("\n7. MOST CONSISTENT ACTIONS (lowest variance):")
    action_stds = np.nanstd(rewards, axis=0)
    consistent = np.argsort(action_stds)
    for rank, a in enumerate(consistent[:5]):
        desc = action_space.describe(int(a))
        print("   action %3d: mean=%.4f std=%.4f | %s" % (
            a, mean_per_action[a], action_stds[a], desc))

    total_time = time.time() - total_start
    print("\nTotal time: %.0fs (%.1f min)" % (total_time, total_time / 60))

    # Save summary as JSON
    summary = {
        "n_actions": int(n_actions),
        "n_cases": len(selected),
        "oracle_reward": float(oracle),
        "oracle_sp": float(best_per_case_sp.mean()),
        "oracle_tc": float(best_per_case_tc.mean()),
        "best_fixed_action": int(top_actions[0]),
        "best_fixed_reward": float(best_fixed),
        "current_agent_reward": current_agent,
        "headroom_fixed": float(best_fixed - current_agent),
        "headroom_oracle": float(oracle - current_agent),
        "mafft_only_oracle": float(mafft_only_oracle),
        "muscle_only_oracle": float(muscle_only_oracle),
        "clustalo_only_oracle": float(clustalo_only_oracle),
        "oracle_tool_distribution": dict(tool_dist),
        "unique_best_actions": unique_best,
        "total_time_s": total_time,
    }
    summary_path = os.path.join(os.path.dirname(__file__), "oracle_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to %s" % summary_path)
    print("Raw results saved to %s" % output_path)


if __name__ == "__main__":
    main()
