"""Sum-of-Pairs (SP) scoring for pairwise alignments.

SP score measures the fraction of correctly aligned residue pairs.
For each pair of residues that are aligned together in the reference,
we check if they are also aligned together in the predicted alignment.
"""

from typing import FrozenSet, Set, Tuple

GAP_CHARS = frozenset("-.~")


def _extract_aligned_pairs(seq1_aligned: str, seq2_aligned: str) -> Set[Tuple[int, int]]:
    """Extract the set of aligned residue-index pairs from a pairwise alignment.

    Walks through alignment columns. For each column where both sequences
    have a residue (non-gap), records (residue_idx_1, residue_idx_2).

    Args:
        seq1_aligned: First sequence with gap characters.
        seq2_aligned: Second sequence with gap characters.

    Returns:
        Set of (idx1, idx2) tuples representing aligned residue pairs.
    """
    pairs = set()
    idx1, idx2 = 0, 0

    for col in range(len(seq1_aligned)):
        is_gap1 = seq1_aligned[col] in GAP_CHARS
        is_gap2 = seq2_aligned[col] in GAP_CHARS

        if not is_gap1 and not is_gap2:
            pairs.add((idx1, idx2))

        if not is_gap1:
            idx1 += 1
        if not is_gap2:
            idx2 += 1

    return pairs


def sp_score(
    pred_seq1: str, pred_seq2: str,
    ref_seq1: str, ref_seq2: str,
) -> float:
    """Compute SP score: fraction of reference pairs found in prediction.

    SP = |ref_pairs ∩ pred_pairs| / |ref_pairs|

    Args:
        pred_seq1: Predicted alignment of sequence 1 (with gaps).
        pred_seq2: Predicted alignment of sequence 2 (with gaps).
        ref_seq1: Reference alignment of sequence 1 (with gaps).
        ref_seq2: Reference alignment of sequence 2 (with gaps).

    Returns:
        SP score in [0, 1]. Returns 0.0 if reference has no aligned pairs.
    """
    ref_pairs = _extract_aligned_pairs(ref_seq1, ref_seq2)
    if not ref_pairs:
        return 0.0

    pred_pairs = _extract_aligned_pairs(pred_seq1, pred_seq2)
    correct = len(ref_pairs & pred_pairs)

    return correct / len(ref_pairs)
