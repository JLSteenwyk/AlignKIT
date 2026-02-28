"""MSA-level Sum-of-Pairs (SP) scoring.

Decomposes the MSA into all N*(N-1)/2 pairwise projections, scores each
pair using the pairwise SP metric, and returns the mean.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from Bio.Align import MultipleSeqAlignment

from data.balibase_parser import _normalize_id, _remove_gap_only_columns, _standardize_gaps
from scoring.sp_score import _extract_aligned_pairs

GAP_CHARS = frozenset("-.~")


def _build_seq_map(msa: MultipleSeqAlignment) -> Dict[str, int]:
    """Map normalized sequence IDs to their index in the MSA."""
    seq_map = {}
    for i, record in enumerate(msa):
        norm_id = _normalize_id(record.id)
        seq_map[norm_id] = i
    return seq_map


def _project_pairwise(msa: MultipleSeqAlignment, idx_i: int, idx_j: int) -> Tuple[str, str]:
    """Project two sequences from an MSA into a pairwise alignment.

    Removes columns where both sequences have gaps.
    """
    seq_i = _standardize_gaps(str(msa[idx_i].seq))
    seq_j = _standardize_gaps(str(msa[idx_j].seq))
    return _remove_gap_only_columns(seq_i, seq_j)


def msa_sp_score(
    pred_msa: MultipleSeqAlignment,
    ref_msa: MultipleSeqAlignment,
) -> float:
    """Compute MSA-level SP score via pairwise decomposition.

    For each pair of sequences present in both pred and ref MSAs:
    1. Project the pair from each MSA (removing double-gap columns)
    2. Compute pairwise SP score
    3. Return mean across all pairs

    Args:
        pred_msa: Predicted MSA (from MAFFT/MUSCLE).
        ref_msa: Reference MSA (from BAliBASE).

    Returns:
        Mean SP score in [0, 1]. Returns 0.0 if no common pairs found.
    """
    pred_map = _build_seq_map(pred_msa)
    ref_map = _build_seq_map(ref_msa)

    # Find common sequence IDs
    common_ids = sorted(set(pred_map.keys()) & set(ref_map.keys()))
    if len(common_ids) < 2:
        return 0.0

    sp_scores = []
    for i in range(len(common_ids)):
        for j in range(i + 1, len(common_ids)):
            id_i = common_ids[i]
            id_j = common_ids[j]

            # Project pairwise from each MSA
            pred_seq_i, pred_seq_j = _project_pairwise(pred_msa, pred_map[id_i], pred_map[id_j])
            ref_seq_i, ref_seq_j = _project_pairwise(ref_msa, ref_map[id_i], ref_map[id_j])

            # Extract aligned pairs and compute SP
            ref_pairs = _extract_aligned_pairs(ref_seq_i, ref_seq_j)
            if not ref_pairs:
                continue

            pred_pairs = _extract_aligned_pairs(pred_seq_i, pred_seq_j)
            correct = len(ref_pairs & pred_pairs)
            sp_scores.append(correct / len(ref_pairs))

    return float(sum(sp_scores) / len(sp_scores)) if sp_scores else 0.0
