"""Combined reward function from SP and TC scores."""


def compute_reward(sp: float, tc: float, sp_weight: float = 0.5, tc_weight: float = 0.5) -> float:
    """Compute weighted combination of SP and TC scores.

    Args:
        sp: Sum-of-Pairs score in [0, 1].
        tc: Total Column score in [0, 1].
        sp_weight: Weight for SP score.
        tc_weight: Weight for TC score.

    Returns:
        Combined reward in [0, 1].
    """
    return sp_weight * sp + tc_weight * tc
