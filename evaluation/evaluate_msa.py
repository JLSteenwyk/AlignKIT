"""Evaluation and baselines for MSA RL parameter learning."""

import subprocess
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from Bio import AlignIO

from agent.continuous_policy import ContinuousPolicy
from agent.reinforce_continuous import ReinforceContinuousAgent
from config import MSAConfig
from data.msa_dataset import MSADataset, MSATestCase, load_msa_test_cases
from env.msa_env import MSAEnv
from scoring.msa_sp_score import msa_sp_score
from scoring.msa_tc_score import msa_tc_score
from scoring.reward import compute_reward
from training.checkpointer import load_checkpoint


def _run_mafft(config, fasta_path, op, ep):
    """Run MAFFT with given op/ep and return MSA or None."""
    cmd = [
        str(config.mafft_bin),
        "--localpair",
        "--maxiterate", "10",
        "--op", str(op),
        "--ep", str(ep),
        "--thread", str(config.mafft_threads),
        "--quiet",
        str(fasta_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=config.mafft_timeout,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return AlignIO.read(StringIO(result.stdout), "fasta")
    except Exception:
        return None


def _score_case(config, case, op, ep):
    """Run MAFFT on a case and return (reward, sp, tc)."""
    pred_msa = _run_mafft(config, case.fasta_path, op, ep)
    if pred_msa is None or len(pred_msa) < 2:
        return 0.0, 0.0, 0.0
    sp = msa_sp_score(pred_msa, case.ref_alignment)
    tc = msa_tc_score(pred_msa, case.ref_alignment)
    r = compute_reward(sp, tc, config.msa_sp_weight, config.msa_tc_weight)
    return r, sp, tc


# ---- Baselines ----

def mafft_default_baseline(eval_dataset, config):
    """MAFFT --localpair --maxiterate 10 with default params (op=1.53, ep=0.0)."""
    op, ep = 1.53, 0.0
    rewards, sps, tcs = [], [], []
    per_set = defaultdict(lambda: {"rewards": [], "sps": [], "tcs": []})

    t0 = time.time()
    for case in eval_dataset:
        r, sp, tc = _score_case(config, case, op, ep)
        rewards.append(r)
        sps.append(sp)
        tcs.append(tc)
        per_set[case.ref_set]["rewards"].append(r)
        per_set[case.ref_set]["sps"].append(sp)
        per_set[case.ref_set]["tcs"].append(tc)
    elapsed = time.time() - t0

    result = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_cases": len(rewards),
        "op": op, "ep": ep,
        "elapsed": elapsed,
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


def best_constant_baseline(eval_dataset, config):
    """Grid search over constant (op, ep) to find best fixed parameters."""
    op_values = [0.5, 1.0, 1.53, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ep_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    cases = list(eval_dataset)
    best_reward = -1
    best_op, best_ep = 1.53, 0.0

    for op in op_values:
        for ep in ep_values:
            rewards = []
            for case in cases:
                r, _, _ = _score_case(config, case, op, ep)
                rewards.append(r)
            mean_r = float(np.mean(rewards))
            if mean_r > best_reward:
                best_reward = mean_r
                best_op, best_ep = op, ep
            print("  grid op=%.2f ep=%.2f -> R=%.4f" % (op, ep, mean_r))

    # Re-evaluate best with full metrics
    rewards, sps, tcs = [], [], []
    for case in cases:
        r, sp, tc = _score_case(config, case, best_op, best_ep)
        rewards.append(r)
        sps.append(sp)
        tcs.append(tc)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sp_mean": float(np.mean(sps)),
        "tc_mean": float(np.mean(tcs)),
        "n_cases": len(rewards),
        "op": best_op, "ep": best_ep,
    }


def per_case_oracle(eval_dataset, config):
    """Per-case oracle: best (op, ep) per case from grid."""
    op_values = [0.5, 1.0, 1.53, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ep_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    best_rewards, best_sps, best_tcs = [], [], []

    for case in eval_dataset:
        best_r = -1
        best_sp, best_tc = 0, 0
        for op in op_values:
            for ep in ep_values:
                r, sp, tc = _score_case(config, case, op, ep)
                if r > best_r:
                    best_r = r
                    best_sp, best_tc = sp, tc
        best_rewards.append(best_r)
        best_sps.append(best_sp)
        best_tcs.append(best_tc)
        print("  oracle %s: R=%.4f" % (case.case_id, best_r))

    return {
        "reward_mean": float(np.mean(best_rewards)),
        "reward_std": float(np.std(best_rewards)),
        "sp_mean": float(np.mean(best_sps)),
        "tc_mean": float(np.mean(best_tcs)),
        "n_cases": len(best_rewards),
    }


# ---- Agent evaluation ----

def evaluate_msa_agent(config: MSAConfig, checkpoint: str, max_cases: int = None):
    """Full evaluation: load agent, run on eval set, compare to baselines."""

    device = torch.device("cpu")

    # Data
    print("Loading evaluation MSA test cases...")
    eval_cases = load_msa_test_cases(config.data_dir, config.eval_ref_sets)
    eval_dataset = MSADataset(eval_cases)

    if len(eval_dataset) == 0:
        print("No evaluation data. Check BAliBASE data path.")
        return

    eval_list = list(eval_dataset)
    if max_cases and len(eval_list) > max_cases:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(eval_list), max_cases, replace=False)
        eval_list = [eval_list[i] for i in indices]
    eval_subset = MSADataset(eval_list)

    # Load agent
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
    load_checkpoint(policy, None, config.checkpoint_dir, checkpoint, device)
    policy.eval()

    env = MSAEnv(eval_subset, config)
    agent = ReinforceContinuousAgent(policy, config, device)

    # Evaluate agent
    print("\n=== RL Agent (greedy) on %d MSAs ===" % len(eval_list))
    rewards, sps, tcs = [], [], []
    per_set = defaultdict(lambda: {"rewards": [], "sps": [], "tcs": []})

    t0 = time.time()
    for case in eval_list:
        state = env.reset_with(case)
        action = agent.select_action_greedy(state)
        reward, info = env.step(action)
        rewards.append(reward)
        sps.append(info["sp"])
        tcs.append(info["tc"])
        per_set[case.ref_set]["rewards"].append(reward)
        per_set[case.ref_set]["sps"].append(info["sp"])
        per_set[case.ref_set]["tcs"].append(info["tc"])

    elapsed = time.time() - t0
    print("  Reward: %.4f +/- %.4f" % (np.mean(rewards), np.std(rewards)))
    print("  SP: %.4f  TC: %.4f" % (np.mean(sps), np.mean(tcs)))
    print("  Time: %.1fs" % elapsed)
    for rs in sorted(per_set.keys()):
        d = per_set[rs]
        print("    %s: R=%.4f SP=%.4f TC=%.4f (n=%d)" % (
            rs, np.mean(d["rewards"]), np.mean(d["sps"]),
            np.mean(d["tcs"]), len(d["rewards"])))

    # MAFFT default baseline
    print("\n=== MAFFT Default (op=1.53, ep=0.0, --localpair --maxiterate 10) ===")
    default_result = mafft_default_baseline(eval_subset, config)
    print("  Reward: %.4f +/- %.4f" % (default_result["reward_mean"], default_result["reward_std"]))
    print("  SP: %.4f  TC: %.4f" % (default_result["sp_mean"], default_result["tc_mean"]))
    for rs, data in sorted(default_result["per_set"].items()):
        print("    %s: R=%.4f SP=%.4f TC=%.4f (n=%d)" % (
            rs, data["reward_mean"], data["sp_mean"],
            data["tc_mean"], data["n_cases"]))

    # Best constant baseline
    print("\n=== Best Constant (grid search over op x ep) ===")
    best_const = best_constant_baseline(eval_subset, config)
    print("  Reward: %.4f +/- %.4f" % (best_const["reward_mean"], best_const["reward_std"]))
    print("  SP: %.4f  TC: %.4f" % (best_const["sp_mean"], best_const["tc_mean"]))
    print("  Best: op=%.2f, ep=%.2f" % (best_const["op"], best_const["ep"]))

    # Per-case oracle
    print("\n=== Per-Case Oracle ===")
    oracle = per_case_oracle(eval_subset, config)
    print("  Reward: %.4f +/- %.4f" % (oracle["reward_mean"], oracle["reward_std"]))
    print("  SP: %.4f  TC: %.4f" % (oracle["sp_mean"], oracle["tc_mean"]))
