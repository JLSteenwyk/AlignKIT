"""CLI entry point: pw-train | pw-evaluate | msa-* commands"""

import argparse
import sys


# =============================================================================
# Pairwise NW commands (new: continuous gap penalty RL)
# =============================================================================

def cmd_pw_train(args):
    """Run pairwise NW gap penalty training."""
    from config import Config
    from training.train_pairwise import train_pairwise

    config = Config()
    if args.episodes:
        config.num_episodes = args.episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.seed:
        config.seed = args.seed
    if args.regions:
        config.num_regions = args.regions
        config.action_dim = 2 * args.regions
    if args.eval_every:
        config.eval_every = args.eval_every

    train_pairwise(config, resume_from=args.resume)


def cmd_pw_evaluate(args):
    """Run pairwise NW evaluation."""
    from config import Config
    from evaluation.evaluate_pairwise import evaluate_pairwise_agent

    config = Config()
    if args.regions:
        config.num_regions = args.regions
        config.action_dim = 2 * args.regions
    checkpoint = args.checkpoint if args.checkpoint else "checkpoint_latest.pt"
    max_pairs = args.max_pairs if args.max_pairs else None
    evaluate_pairwise_agent(config, checkpoint, max_pairs=max_pairs)


# =============================================================================
# MSA commands (new)
# =============================================================================

def cmd_msa_train(args):
    """Run MSA training with continuous MAFFT parameters."""
    from config import MSAConfig
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
    if args.eval_every:
        config.eval_every = args.eval_every

    train_msa(config, resume_from=args.resume)


def cmd_msa_evaluate(args):
    """Run MSA evaluation."""
    from config import MSAConfig
    from evaluation.evaluate_msa import evaluate_msa_agent

    config = MSAConfig()
    checkpoint = args.checkpoint if args.checkpoint else "checkpoint_latest.pt"
    max_cases = args.max_cases if args.max_cases else None
    evaluate_msa_agent(config, checkpoint, max_cases=max_cases)


# =============================================================================
# CLI setup
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL Framework for Sequence Alignment Parameter Optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Pairwise NW commands (continuous gap penalty RL) ---
    pw_train_parser = subparsers.add_parser("pw-train", help="Train pairwise NW gap penalty agent")
    pw_train_parser.add_argument("--episodes", type=int, help="Number of episodes")
    pw_train_parser.add_argument("--batch-size", type=int, help="Batch size")
    pw_train_parser.add_argument("--lr", type=float, help="Learning rate")
    pw_train_parser.add_argument("--seed", type=int, help="Random seed")
    pw_train_parser.add_argument("--regions", type=int, help="Number of regions K (default 3)")
    pw_train_parser.add_argument("--resume", type=str, default=None,
                                 help="Resume from checkpoint file")
    pw_train_parser.add_argument("--eval-every", type=int, help="Evaluate every N episodes (default 500)")

    pw_eval_parser = subparsers.add_parser("pw-evaluate", help="Evaluate pairwise NW agent")
    pw_eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint filename")
    pw_eval_parser.add_argument("--max-pairs", type=int, help="Max eval pairs")
    pw_eval_parser.add_argument("--regions", type=int, help="Number of regions K (default 3)")

    # --- MSA commands (continuous MAFFT parameter RL) ---
    msa_train_parser = subparsers.add_parser("msa-train", help="Train MSA RL agent (continuous op/ep)")
    msa_train_parser.add_argument("--episodes", type=int, help="Number of episodes")
    msa_train_parser.add_argument("--batch-size", type=int, help="Batch size")
    msa_train_parser.add_argument("--lr", type=float, help="Learning rate")
    msa_train_parser.add_argument("--seed", type=int, help="Random seed")
    msa_train_parser.add_argument("--resume", type=str, default=None,
                                  help="Resume from checkpoint file (e.g. checkpoint_latest.pt)")
    msa_train_parser.add_argument("--eval-every", type=int, default=None,
                                  help="Evaluate every N episodes (default: 200)")

    msa_eval_parser = subparsers.add_parser("msa-evaluate", help="Evaluate MSA agent")
    msa_eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint filename")
    msa_eval_parser.add_argument("--max-cases", type=int, help="Max eval MSAs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "pw-train": cmd_pw_train,
        "pw-evaluate": cmd_pw_evaluate,
        "msa-train": cmd_msa_train,
        "msa-evaluate": cmd_msa_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
