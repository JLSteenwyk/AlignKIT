"""Main training loop for the MSA REINFORCE agent."""

import random
import time
from collections import Counter

import numpy as np
import torch

from config_msa import MSAConfig
from data.msa_dataset import build_msa_datasets
from env.msa_action_space import MSAActionSpace
from env.msa_alignment_env import MSAAlignmentEnv
from training.checkpointer import save_checkpoint, load_checkpoint
from training.logger import JSONLLogger


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_policy_and_agent(config: MSAConfig, action_space: MSAActionSpace, device: torch.device):
    """Construct policy and agent based on config.policy_type."""
    if config.policy_type == "hierarchical":
        from agent.hierarchical_policy import HierarchicalPolicy
        from agent.reinforce_hierarchical import HierarchicalReinforceAgent

        policy = HierarchicalPolicy(
            input_dim=config.input_dim,
            hidden_sizes=config.hidden_sizes,
            param_sizes=action_space.param_sizes,
        )
        agent = HierarchicalReinforceAgent(policy, config, device)
    else:
        from agent.policy_network import PolicyNetwork
        from agent.reinforce_flat_msa import ReinforceFlatMSAAgent

        policy = PolicyNetwork(
            input_dim=config.input_dim,
            hidden_sizes=config.hidden_sizes,
            output_dim=config.num_actions,
        )
        agent = ReinforceFlatMSAAgent(policy, config, device)

    return policy, agent


def train_msa(config: MSAConfig, resume_from: str = None):
    """Run the full REINFORCE training loop for MSA tool selection.

    1. Load BAliBASE MSA test cases
    2. Create environment, agent
    3. Collect batches of episodes, update policy
    4. Log metrics (including tool distribution), checkpoint, evaluate

    Args:
        resume_from: Optional checkpoint filename to resume training from.
    """
    config.ensure_dirs()
    set_seeds(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hierarchical = config.policy_type == "hierarchical"

    # Data
    benchmark_dir = config.benchmark_dir if config.use_benchmarks else None
    train_dataset, eval_dataset = build_msa_datasets(
        config.data_dir, config.all_ref_sets, config.split_ratio, config.split_seed,
        benchmark_dir=benchmark_dir,
        subsample_benchmarks_to=config.subsample_benchmarks_to,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training data loaded. Check BAliBASE data path.")

    # Environment + action space
    action_space = MSAActionSpace(config)
    env = MSAAlignmentEnv(train_dataset, action_space, config)
    eval_env = MSAAlignmentEnv(eval_dataset, action_space, config)

    # Agent
    policy, agent = _build_policy_and_agent(config, action_space, device)

    # Resume from checkpoint if requested
    start_episode = 0
    if resume_from is not None:
        ckpt = load_checkpoint(policy, agent.optimizer, config.checkpoint_dir, resume_from, device)
        start_episode = ckpt.get("episode", 0)
        if "baseline" in ckpt:
            agent._baseline = ckpt["baseline"]
            agent._baseline_initialized = True
        print(f"Resumed from checkpoint '{resume_from}' at episode {start_episode}")

    # Logging
    train_logger = JSONLLogger(config.log_dir, "msa_train")
    eval_logger = JSONLLogger(config.log_dir, "msa_eval")

    total_episodes = start_episode + config.num_episodes
    policy_label = "hierarchical" if hierarchical else "flat"
    if hierarchical:
        sizes = action_space.param_sizes
        print(f"\nPolicy: {policy_label} — tool head (3-way) + param heads ({sizes[0]}/{sizes[1]}/{sizes[2]})")
    else:
        print(f"\nPolicy: {policy_label} — flat softmax ({config.num_actions} actions)")
    print(f"Starting MSA training: {config.num_episodes} episodes (ep {start_episode} -> {total_episodes}), batch_size={config.batch_size}")
    print(f"Action space: {config.num_actions} actions (MAFFT: {action_space.n_mafft}, MUSCLE: {action_space.n_muscle}, ClustalO: {action_space.n_clustalo})")
    print(f"Train MSAs: {len(train_dataset)}, Eval MSAs: {len(eval_dataset)}")
    print()

    # Training loop
    episode = start_episode
    total_start = time.time()

    # Running stats
    recent_rewards = []
    recent_sp = []
    recent_tc = []
    recent_tools = []
    recent_elapsed = []
    recent_errors = 0

    while episode < total_episodes:
        # Collect a batch of episodes
        if hierarchical:
            batch = {"states": [], "tool_indices": [], "param_indices": [], "rewards": []}
        else:
            batch = {"states": [], "actions": [], "rewards": []}
        batch_infos = []

        for _ in range(config.batch_size):
            state = env.reset()

            if hierarchical:
                tool_idx, param_idx, log_prob, tool_ent, param_ent = agent.select_action(state)
                reward, info = env.step_hierarchical(tool_idx, param_idx)
                batch["states"].append(state)
                batch["tool_indices"].append(tool_idx)
                batch["param_indices"].append(param_idx)
                batch["rewards"].append(reward)
            else:
                action, log_prob, entropy = agent.select_action(state)
                reward, info = env.step(action)
                batch["states"].append(state)
                batch["actions"].append(action)
                batch["rewards"].append(reward)

            batch_infos.append(info)

            recent_rewards.append(reward)
            recent_sp.append(info.get("sp", 0.0))
            recent_tc.append(info.get("tc", 0.0))
            recent_tools.append(info.get("tool", "unknown"))
            recent_elapsed.append(info.get("elapsed", 0.0))
            if "error" in info:
                recent_errors += 1

        # Update policy
        update_metrics = agent.update(batch)
        episode += config.batch_size

        # Log
        if episode % config.log_every < config.batch_size:
            window = config.log_every
            avg_reward = np.mean(recent_rewards[-window:])
            avg_sp = np.mean(recent_sp[-window:])
            avg_tc = np.mean(recent_tc[-window:])
            avg_elapsed = np.mean(recent_elapsed[-window:])

            # Tool distribution in recent window
            tool_window = recent_tools[-window:]
            tool_counts = Counter(tool_window)
            n_tw = len(tool_window) if tool_window else 1
            mafft_frac = tool_counts.get("mafft", 0) / n_tw
            muscle_frac = tool_counts.get("muscle", 0) / n_tw
            clustalo_frac = tool_counts.get("clustalo", 0) / n_tw

            metrics = {
                "reward_mean": float(avg_reward),
                "sp_mean": float(avg_sp),
                "tc_mean": float(avg_tc),
                "mafft_frac": float(mafft_frac),
                "muscle_frac": float(muscle_frac),
                "clustalo_frac": float(clustalo_frac),
                "avg_elapsed": float(avg_elapsed),
                "error_count": recent_errors,
                **update_metrics,
            }
            train_logger.log(metrics, episode)

            elapsed = time.time() - total_start
            if hierarchical:
                print(
                    f"Ep {episode:5d} | "
                    f"R={avg_reward:.4f} SP={avg_sp:.4f} TC={avg_tc:.4f} | "
                    f"MAFFT={mafft_frac:.0%} MUSCLE={muscle_frac:.0%} ClustalO={clustalo_frac:.0%} | "
                    f"toolH={update_metrics['tool_entropy']:.3f} "
                    f"paramH={update_metrics['param_entropy']:.3f} "
                    f"t={avg_elapsed:.1f}s | "
                    f"{elapsed:.0f}s"
                )
            else:
                print(
                    f"Ep {episode:5d} | "
                    f"R={avg_reward:.4f} SP={avg_sp:.4f} TC={avg_tc:.4f} | "
                    f"MAFFT={mafft_frac:.0%} MUSCLE={muscle_frac:.0%} ClustalO={clustalo_frac:.0%} | "
                    f"Ent={update_metrics['entropy']:.3f} "
                    f"vL={update_metrics.get('value_loss', 0):.4f} "
                    f"t={avg_elapsed:.1f}s | "
                    f"{elapsed:.0f}s"
                )
            recent_errors = 0

        # Checkpoint
        if episode % config.checkpoint_every < config.batch_size:
            save_checkpoint(policy, agent.optimizer, episode, config.checkpoint_dir,
                            extra={"baseline": agent._baseline})

        # Evaluate
        if episode % config.eval_every < config.batch_size and len(eval_dataset) > 0:
            eval_metrics = _evaluate_msa(agent, eval_env, eval_dataset, action_space,
                                         hierarchical=hierarchical)
            eval_logger.log(eval_metrics, episode)
            print(
                f"  [EVAL] R={eval_metrics['reward_mean']:.4f} "
                f"SP={eval_metrics['sp_mean']:.4f} TC={eval_metrics['tc_mean']:.4f} "
                f"MAFFT={eval_metrics['mafft_frac']:.0%} "
                f"MUSCLE={eval_metrics['muscle_frac']:.0%} "
                f"ClustalO={eval_metrics['clustalo_frac']:.0%} "
                f"(n={eval_metrics['n_cases']})"
            )

    # Final save
    save_checkpoint(policy, agent.optimizer, episode, config.checkpoint_dir, "checkpoint_final.pt",
                    extra={"baseline": agent._baseline})
    total_time = time.time() - total_start
    print(f"\nMSA training complete: {episode} episodes in {total_time:.1f}s")


def _evaluate_msa(agent, env, dataset, action_space, max_cases=None, hierarchical=False):
    """Run greedy evaluation on the eval MSA dataset."""
    rewards, sps, tcs = [], [], []
    tools = []
    per_set = {}

    cases = list(dataset)
    if max_cases and len(cases) > max_cases:
        cases = random.sample(cases, max_cases)

    for case in cases:
        state = env.reset_with_case(case)

        if hierarchical:
            tool_idx, param_idx = agent.select_action_greedy(state)
            reward, info = env.step_hierarchical(tool_idx, param_idx)
        else:
            action = agent.select_action_greedy(state)
            reward, info = env.step(action)

        rewards.append(reward)
        sps.append(info.get("sp", 0.0))
        tcs.append(info.get("tc", 0.0))
        tools.append(info.get("tool", "unknown"))

        rs = info.get("ref_set", "unknown")
        if rs not in per_set:
            per_set[rs] = {"rewards": [], "sps": [], "tcs": []}
        per_set[rs]["rewards"].append(reward)
        per_set[rs]["sps"].append(info.get("sp", 0.0))
        per_set[rs]["tcs"].append(info.get("tc", 0.0))

    tool_counts = Counter(tools)
    n = len(tools)

    result = {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "sp_mean": float(np.mean(sps)) if sps else 0.0,
        "tc_mean": float(np.mean(tcs)) if tcs else 0.0,
        "mafft_frac": float(tool_counts.get("mafft", 0) / n) if n > 0 else 0.0,
        "muscle_frac": float(tool_counts.get("muscle", 0) / n) if n > 0 else 0.0,
        "clustalo_frac": float(tool_counts.get("clustalo", 0) / n) if n > 0 else 0.0,
        "n_cases": len(rewards),
    }

    for rs, data in per_set.items():
        result[f"{rs}_reward"] = float(np.mean(data["rewards"]))
        result[f"{rs}_sp"] = float(np.mean(data["sps"]))
        result[f"{rs}_tc"] = float(np.mean(data["tcs"]))

    return result
