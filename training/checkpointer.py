"""Model checkpoint save/load utilities."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    checkpoint_dir: Path,
    filename: Optional[str] = None,
    extra: Optional[dict] = None,
):
    """Save model and optimizer state.

    Args:
        policy: The policy network (any nn.Module).
        optimizer: The optimizer.
        episode: Current episode number.
        checkpoint_dir: Directory to save checkpoints.
        filename: Optional custom filename.
        extra: Optional dict of additional state to save (e.g. baseline).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_ep{episode}.pt"

    path = checkpoint_dir / filename
    data = {
        "episode": episode,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra is not None:
        data.update(extra)
    torch.save(data, path)
    # Also save as "latest"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(data, latest_path)


def load_checkpoint(
    policy: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: Path,
    filename: str = "checkpoint_latest.pt",
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load model and optimizer state from checkpoint.

    Args:
        policy: The policy network to load weights into (any nn.Module).
        optimizer: The optimizer to load state into (optional).
        checkpoint_dir: Directory containing checkpoints.
        filename: Checkpoint filename.
        device: Device to map tensors to.

    Returns:
        The full checkpoint dict (keys: episode, policy_state_dict,
        optimizer_state_dict, and any extra state like baseline).
    """
    path = checkpoint_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    load_kwargs = {"map_location": device}
    # weights_only requires torch >= 1.13
    import inspect
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(path, **load_kwargs)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
