"""Total Column (TC) scoring for pairwise alignments.

TC score measures the fraction of correctly aligned columns.
For pairwise alignments, a "column" is a match column where both
sequences have residues. TC checks if the same residue-index columns
appear in both reference and prediction.
"""

from typing import List, Set, Tuple

GAP_CHARS = frozenset("-.~")


def _extract_column_signatures(
    seq1_aligned: str, seq2_aligned: str
) -> Set[Tuple[int, int]]:
    """Extract match-column signatures from a pairwise alignment.

    For each column where both sequences have a residue (non-gap),
    record (residue_idx_1, residue_idx_2) as the column signature.

    Note: For pairwise alignments, this is equivalent to the aligned-pairs
    extraction in SP scoring. The distinction matters more for MSA TC scoring
    where entire columns (across all sequences) must match.

    Args:
        seq1_aligned: First sequence with gap characters.
        seq2_aligned: Second sequence with gap characters.

    Returns:
        Set of (idx1, idx2) tuples representing match columns.
    """
    columns = set()
    idx1, idx2 = 0, 0

    for col in range(len(seq1_aligned)):
        is_gap1 = seq1_aligned[col] in GAP_CHARS
        is_gap2 = seq2_aligned[col] in GAP_CHARS

        if not is_gap1 and not is_gap2:
            columns.add((idx1, idx2))

        if not is_gap1:
            idx1 += 1
        if not is_gap2:
            idx2 += 1

    return columns


def tc_score(
    pred_seq1: str, pred_seq2: str,
    ref_seq1: str, ref_seq2: str,
) -> float:
    """Compute TC score: fraction of reference columns found in prediction.

    TC = |ref_columns & pred_columns| / |ref_columns|

    For pairwise alignments, this is numerically equivalent to SP score,
    but the conceptual distinction is educational: SP focuses on residue pairs
    while TC focuses on alignment columns. The difference emerges in MSA scoring.

    Args:
        pred_seq1: Predicted alignment of sequence 1 (with gaps).
        pred_seq2: Predicted alignment of sequence 2 (with gaps).
        ref_seq1: Reference alignment of sequence 1 (with gaps).
        ref_seq2: Reference alignment of sequence 2 (with gaps).

    Returns:
        TC score in [0, 1]. Returns 0.0 if reference has no match columns.
    """
    ref_columns = _extract_column_signatures(ref_seq1, ref_seq2)
    if not ref_columns:
        return 0.0

    pred_columns = _extract_column_signatures(pred_seq1, pred_seq2)
    correct = len(ref_columns & pred_columns)

    return correct / len(ref_columns)
