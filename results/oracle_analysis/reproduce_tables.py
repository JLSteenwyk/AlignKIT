"""Reproduce oracle analysis tables from saved results.

Run: python results/oracle_analysis/reproduce_tables.py
"""
import sys, os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config_msa import MSAConfig
from env.msa_action_space import MSAActionSpace


def main():
    config = MSAConfig()
    action_space = MSAActionSpace(config)
    n_mafft = action_space.n_mafft
    n_muscle = action_space.n_muscle
    mafft_end = n_mafft
    muscle_end = n_mafft + n_muscle
    n_actions = action_space.n_actions

    data_path = os.path.join(os.path.dirname(__file__), "oracle_results.npz")
    data = np.load(data_path, allow_pickle=True)
    rewards = data["rewards"]
    sp = data["sp"]
    tc = data["tc"]
    case_ref_sets = list(data["case_ref_sets"])
    case_ids = list(data["case_ids"])

    # ================================================================
    # Table 1: Per-tool oracle (best config within each tool, per case)
    # ================================================================
    print("=" * 90)
    print("TABLE 1: Per-Database, Per-Tool Oracle Analysis (BAliBASE eval, N=%d)" % len(case_ref_sets))
    print("=" * 90)
    print()
    print("%-8s %4s | %7s %7s %7s | %7s %7s | %7s %s" % (
        "Dataset", "N", "MAFFT", "MUSCLE", "ClstlO", "Oracle", "BstFix", "Headrm", "OrcTool"))
    print("-" * 90)

    all_datasets = sorted(set(case_ref_sets))
    for rs in all_datasets:
        mask = [j for j, r in enumerate(case_ref_sets) if r == rs]
        n = len(mask)
        rs_rewards = rewards[mask]

        mafft_oracle = np.nanmean(np.nanmax(rs_rewards[:, :mafft_end], axis=1))
        muscle_oracle = np.nanmean(np.nanmax(rs_rewards[:, mafft_end:muscle_end], axis=1))
        clustalo_oracle = np.nanmean(np.nanmax(rs_rewards[:, muscle_end:], axis=1))

        oracle = np.nanmean(np.nanmax(rs_rewards, axis=1))
        best_fixed = np.nanmax(np.nanmean(rs_rewards, axis=0))
        headroom = oracle - best_fixed

        best_per_case = np.nanargmax(rs_rewards, axis=1)
        tools = []
        for a in best_per_case:
            if a < mafft_end: tools.append("M")
            elif a < muscle_end: tools.append("U")
            else: tools.append("C")
        tc_count = Counter(tools)
        tool_str = "M=%d U=%d C=%d" % (tc_count.get("M", 0), tc_count.get("U", 0), tc_count.get("C", 0))

        print("%-8s %4d | %7.4f %7.4f %7.4f | %7.4f %7.4f | %+7.4f  %s" % (
            rs, n, mafft_oracle, muscle_oracle, clustalo_oracle, oracle, best_fixed, headroom, tool_str))

    print("-" * 90)

    # Overall
    n = len(case_ref_sets)
    mafft_oracle = np.nanmean(np.nanmax(rewards[:, :mafft_end], axis=1))
    muscle_oracle = np.nanmean(np.nanmax(rewards[:, mafft_end:muscle_end], axis=1))
    clustalo_oracle = np.nanmean(np.nanmax(rewards[:, muscle_end:], axis=1))
    oracle = np.nanmean(np.nanmax(rewards, axis=1))
    best_fixed = np.nanmax(np.nanmean(rewards, axis=0))
    headroom = oracle - best_fixed

    print("%-8s %4d | %7.4f %7.4f %7.4f | %7.4f %7.4f | %+7.4f" % (
        "ALL", n, mafft_oracle, muscle_oracle, clustalo_oracle, oracle, best_fixed, headroom))

    # ================================================================
    # Table 2: Decomposing the oracle gain
    # ================================================================
    print()
    print("=" * 90)
    print("TABLE 2: Decomposing Oracle Gain")
    print("=" * 90)
    print()

    mafft_best_fixed = np.nanmax(np.nanmean(rewards[:, :mafft_end], axis=0))
    muscle_best_fixed = np.nanmax(np.nanmean(rewards[:, mafft_end:muscle_end], axis=0))
    clustalo_best_fixed = np.nanmax(np.nanmean(rewards[:, muscle_end:], axis=0))

    print("Component                               Reward")
    print("-" * 50)
    print("MAFFT best fixed config:                %.4f" % mafft_best_fixed)
    print("MUSCLE best fixed config:               %.4f" % muscle_best_fixed)
    print("ClustalO best fixed config:             %.4f" % clustalo_best_fixed)
    print("Overall best fixed config:              %.4f" % best_fixed)
    print()
    print("MAFFT per-case oracle:                  %.4f  (+%.4f over MAFFT fixed)" % (
        mafft_oracle, mafft_oracle - mafft_best_fixed))
    print("MUSCLE per-case oracle:                 %.4f  (+%.4f over MUSCLE fixed)" % (
        muscle_oracle, muscle_oracle - muscle_best_fixed))
    print("ClustalO per-case oracle:               %.4f  (+%.4f over ClustalO fixed)" % (
        clustalo_oracle, clustalo_oracle - clustalo_best_fixed))
    print()
    print("Full oracle (any tool, any config):     %.4f" % oracle)
    print("Gain from MAFFT param tuning:           +%.4f  (%.1f%% of total headroom)" % (
        mafft_oracle - mafft_best_fixed,
        100 * (mafft_oracle - mafft_best_fixed) / headroom if headroom > 0 else 0))
    print("Gain from cross-tool switching:         +%.4f  (%.1f%% of total headroom)" % (
        oracle - mafft_oracle,
        100 * (oracle - mafft_oracle) / headroom if headroom > 0 else 0))
    print("Total headroom (oracle - best fixed):   +%.4f" % headroom)

    # ================================================================
    # Table 3: Top 10 most common oracle actions
    # ================================================================
    print()
    print("=" * 90)
    print("TABLE 3: Most Common Oracle-Best Actions")
    print("=" * 90)
    print()

    best_per_case_idx = np.nanargmax(rewards, axis=1)
    action_counts = Counter(int(x) for x in best_per_case_idx)
    print("Action  Tool      Count  AvgR    Description")
    print("-" * 80)
    for a, count in action_counts.most_common(10):
        desc = action_space.describe(a)
        tool = action_space.decode(a).tool
        avg_r = np.nanmean(rewards[:, a])
        print("  %3d   %-8s  %3d    %.4f  %s" % (a, tool, count, avg_r, desc))

    # ================================================================
    # Summary statistics
    # ================================================================
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print("Dataset: BAliBASE (6 reference sets: RV11-RV50)")
    print("Eval cases: %d (stratified subsample, up to 8 per ref set)" % len(case_ref_sets))
    print("Action space: %d actions (MAFFT: %d, MUSCLE: %d, ClustalO: %d)" % (
        n_actions, n_mafft, n_muscle, n_actions - n_mafft - n_muscle))
    print("Total alignments evaluated: %d" % (len(case_ref_sets) * n_actions))
    print()
    print("Key finding: MAFFT is the oracle-best tool on %d/%d cases (%.0f%%)." % (
        sum(1 for a in best_per_case_idx if a < mafft_end), len(case_ref_sets),
        100 * sum(1 for a in best_per_case_idx if a < mafft_end) / len(case_ref_sets)))
    print("MAFFT per-case param tuning accounts for %.1f%% of total oracle headroom." % (
        100 * (mafft_oracle - mafft_best_fixed) / headroom if headroom > 0 else 0))
    print("Cross-tool switching adds only %.1f%% additional gain." % (
        100 * (oracle - mafft_oracle) / headroom if headroom > 0 else 0))
    print("Conclusion: A MAFFT-only parameter policy captures nearly all learnable reward.")


if __name__ == "__main__":
    main()
