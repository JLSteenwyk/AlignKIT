"""CLI entry point: train | evaluate | visualize | baseline | msa-* commands"""

import argparse
import sys

from config import Config


# =============================================================================
# Pairwise commands (existing)
# =============================================================================

def cmd_train(args):
    """Run pairwise training."""
    from training.train import train

    config = Config()
    if args.episodes:
        config.num_episodes = args.episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.seed:
        config.seed = args.seed

    train(config)


def cmd_evaluate(args):
    """Run pairwise evaluation."""
    from evaluation.evaluate import evaluate_agent

    config = Config()
    checkpoint = args.checkpoint if args.checkpoint else "checkpoint_latest.pt"
    max_pairs = args.max_pairs if args.max_pairs else 500
    evaluate_agent(config, checkpoint, max_pairs=max_pairs)


def cmd_visualize(args):
    """Generate pairwise visualization plots."""
    from visualization.plots import generate_all_plots

    config = Config()
    generate_all_plots(config)


def cmd_baseline(args):
    """Run pairwise baseline evaluations."""
    from data.dataset import build_datasets
    from env.action_space import ActionSpace
    from evaluation.baselines import default_baseline, grid_search_oracle, per_set_best_fixed

    config = Config()
    _, eval_dataset, _ = build_datasets(
        config.data_dir, config.train_ref_sets, config.eval_ref_sets, config.max_seq_length
    )
    action_space = ActionSpace(config)

    if len(eval_dataset) == 0:
        print("No evaluation data. Run setup.sh first to download BAliBASE.")
        return

    print("\n=== Default Baseline (BLOSUM62, go=-10, ge=-0.5) ===")
    default_result = default_baseline(eval_dataset, action_space, config)
    print(f"  Reward: {default_result['reward_mean']:.4f} +/- {default_result['reward_std']:.4f}")
    print(f"  SP: {default_result['sp_mean']:.4f}  TC: {default_result['tc_mean']:.4f}")
    print(f"  Action: go={default_result['gap_open']}, ge={default_result['gap_extend']}, mat={default_result['matrix']}")

    if not args.skip_oracle:
        print("\n=== Grid Search Oracle (upper bound) ===")
        oracle_result = grid_search_oracle(eval_dataset, action_space, config, max_pairs=args.oracle_pairs)
        print(f"  Reward: {oracle_result['reward_mean']:.4f} +/- {oracle_result['reward_std']:.4f}")
        print(f"  SP: {oracle_result['sp_mean']:.4f}  TC: {oracle_result['tc_mean']:.4f}")

    if not args.skip_per_set:
        print("\n=== Per-Set Best Fixed Action ===")
        per_set_result = per_set_best_fixed(eval_dataset, action_space, config)
        for rs, data in sorted(per_set_result["per_set"].items()):
            print(f"  {rs}: R={data['reward_mean']:.4f} | go={data['gap_open']}, ge={data['gap_extend']}, mat={data['matrix']}")


# =============================================================================
# MSA commands (new)
# =============================================================================

def cmd_msa_train(args):
    """Run MSA training."""
    from config_msa import MSAConfig
    from training.train_msa import train_msa

    config = MSAConfig()
    if args.episodes:
        config.num_episodes = args.episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.seed:
        config.seed = args.seed

    train_msa(config)


def cmd_msa_evaluate(args):
    """Run MSA evaluation."""
    from config_msa import MSAConfig
    from evaluation.evaluate_msa import evaluate_msa_agent

    config = MSAConfig()
    checkpoint = args.checkpoint if args.checkpoint else "checkpoint_latest.pt"
    max_cases = args.max_cases if args.max_cases else None
    evaluate_msa_agent(config, checkpoint, max_cases=max_cases)


def cmd_msa_baseline(args):
    """Run MSA baseline evaluations (MAFFT default + MUSCLE default + ClustalO default)."""
    from config_msa import MSAConfig
    from data.msa_dataset import build_msa_datasets
    from env.msa_action_space import MSAActionSpace
    from evaluation.msa_baselines import (
        mafft_default_baseline, muscle_default_baseline, clustalo_default_baseline,
    )

    config = MSAConfig()
    benchmark_dir = config.benchmark_dir if config.use_benchmarks else None
    _, eval_dataset = build_msa_datasets(
        config.data_dir, config.all_ref_sets, config.split_ratio, config.split_seed,
        benchmark_dir=benchmark_dir,
    )

    if len(eval_dataset) == 0:
        print("No MSA evaluation data. Check BAliBASE data path.")
        return

    action_space = MSAActionSpace(config)

    print("\n=== MAFFT Default Baseline ===")
    mafft_result = mafft_default_baseline(eval_dataset, action_space, config)
    print(f"  Reward: {mafft_result['reward_mean']:.4f} +/- {mafft_result['reward_std']:.4f}")
    print(f"  SP: {mafft_result['sp_mean']:.4f}  TC: {mafft_result['tc_mean']:.4f}")
    print(f"  Action: {mafft_result['action_desc']}")
    print(f"  Avg time: {mafft_result['avg_elapsed']:.1f}s ({mafft_result['n_cases']} cases)")
    for rs, data in sorted(mafft_result["per_set"].items()):
        print(f"    {rs}: R={data['reward_mean']:.4f} SP={data['sp_mean']:.4f} TC={data['tc_mean']:.4f} (n={data['n_cases']})")

    print("\n=== MUSCLE Default Baseline ===")
    muscle_result = muscle_default_baseline(eval_dataset, action_space, config)
    print(f"  Reward: {muscle_result['reward_mean']:.4f} +/- {muscle_result['reward_std']:.4f}")
    print(f"  SP: {muscle_result['sp_mean']:.4f}  TC: {muscle_result['tc_mean']:.4f}")
    print(f"  Action: {muscle_result['action_desc']}")
    print(f"  Avg time: {muscle_result['avg_elapsed']:.1f}s ({muscle_result['n_cases']} cases)")
    for rs, data in sorted(muscle_result["per_set"].items()):
        print(f"    {rs}: R={data['reward_mean']:.4f} SP={data['sp_mean']:.4f} TC={data['tc_mean']:.4f} (n={data['n_cases']})")

    print("\n=== Clustal Omega Default Baseline ===")
    clustalo_result = clustalo_default_baseline(eval_dataset, action_space, config)
    print(f"  Reward: {clustalo_result['reward_mean']:.4f} +/- {clustalo_result['reward_std']:.4f}")
    print(f"  SP: {clustalo_result['sp_mean']:.4f}  TC: {clustalo_result['tc_mean']:.4f}")
    print(f"  Action: {clustalo_result['action_desc']}")
    print(f"  Avg time: {clustalo_result['avg_elapsed']:.1f}s ({clustalo_result['n_cases']} cases)")
    for rs, data in sorted(clustalo_result["per_set"].items()):
        print(f"    {rs}: R={data['reward_mean']:.4f} SP={data['sp_mean']:.4f} TC={data['tc_mean']:.4f} (n={data['n_cases']})")


def cmd_msa_visualize(args):
    """Generate MSA visualization plots."""
    from config_msa import MSAConfig
    from visualization.msa_plots import generate_msa_plots

    config = MSAConfig()
    generate_msa_plots(config)


# =============================================================================
# CLI setup
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL Framework for Sequence Alignment Parameter Optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Pairwise commands ---
    train_parser = subparsers.add_parser("train", help="Train pairwise RL agent")
    train_parser.add_argument("--episodes", type=int, help="Number of episodes")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--seed", type=int, help="Random seed")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate pairwise agent")
    eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint filename")
    eval_parser.add_argument("--max-pairs", type=int, help="Max eval pairs (default 500)")

    subparsers.add_parser("visualize", help="Generate pairwise plots")

    baseline_parser = subparsers.add_parser("baseline", help="Run pairwise baselines")
    baseline_parser.add_argument("--skip-oracle", action="store_true", help="Skip grid search oracle")
    baseline_parser.add_argument("--skip-per-set", action="store_true", help="Skip per-set best fixed")
    baseline_parser.add_argument("--oracle-pairs", type=int, default=50, help="Max pairs for oracle")

    # --- MSA commands ---
    msa_train_parser = subparsers.add_parser("msa-train", help="Train MSA RL agent")
    msa_train_parser.add_argument("--episodes", type=int, help="Number of episodes")
    msa_train_parser.add_argument("--batch-size", type=int, help="Batch size")
    msa_train_parser.add_argument("--lr", type=float, help="Learning rate")
    msa_train_parser.add_argument("--seed", type=int, help="Random seed")

    msa_eval_parser = subparsers.add_parser("msa-evaluate", help="Evaluate MSA agent")
    msa_eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint filename")
    msa_eval_parser.add_argument("--max-cases", type=int, help="Max eval MSAs")

    subparsers.add_parser("msa-baseline", help="Run MSA baselines (MAFFT/MUSCLE/ClustalO defaults)")

    subparsers.add_parser("msa-visualize", help="Generate MSA plots")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "baseline": cmd_baseline,
        "msa-train": cmd_msa_train,
        "msa-evaluate": cmd_msa_evaluate,
        "msa-baseline": cmd_msa_baseline,
        "msa-visualize": cmd_msa_visualize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
