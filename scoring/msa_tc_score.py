"""MSA-level Total Column (TC) scoring.

TC score measures the fraction of reference alignment columns that are
exactly reproduced in the predicted alignment. A column is defined by
the set of (seq_id, residue_index) pairs for all non-gap positions.
"""

from typing import Dict, FrozenSet, Set, Tuple

from Bio.Align import MultipleSeqAlignment

from data.balibase_parser import _normalize_id, _standardize_gaps

GAP_CHARS = frozenset("-.~")


def _extract_column_signatures(
    msa: MultipleSeqAlignment,
) -> Set[FrozenSet[Tuple[str, int]]]:
    """Extract column signatures from an MSA.

    For each column with 2+ non-gap residues, build a signature:
    frozenset of (normalized_seq_id, residue_index) tuples.

    Args:
        msa: A MultipleSeqAlignment object.

    Returns:
        Set of frozensets, each representing a column signature.
    """
    n_seqs = len(msa)
    if n_seqs == 0:
        return set()

    alignment_len = msa.get_alignment_length()

    # Build normalized ID list and standardized sequences
    ids = [_normalize_id(msa[i].id) for i in range(n_seqs)]
    seqs = [_standardize_gaps(str(msa[i].seq)) for i in range(n_seqs)]

    # Track residue index per sequence
    residue_indices = [0] * n_seqs

    columns = set()
    for col in range(alignment_len):
        signature = []
        for s in range(n_seqs):
            if col < len(seqs[s]) and seqs[s][col] not in GAP_CHARS:
                signature.append((ids[s], residue_indices[s]))
                residue_indices[s] += 1
            elif col < len(seqs[s]) and seqs[s][col] in GAP_CHARS:
                pass  # gap, don't increment
            # else: beyond sequence length, treat as gap

        # Only include columns with 2+ aligned residues
        if len(signature) >= 2:
            columns.add(frozenset(signature))

    return columns


def msa_tc_score(
    pred_msa: MultipleSeqAlignment,
    ref_msa: MultipleSeqAlignment,
) -> float:
    """Compute MSA-level TC score.

    TC = |ref_columns intersection pred_columns| / |ref_columns|

    Only considers sequences present in both MSAs. Column signatures
    are built from (seq_id, residue_index) tuples for non-gap positions.

    Args:
        pred_msa: Predicted MSA (from MAFFT/MUSCLE).
        ref_msa: Reference MSA (from BAliBASE).

    Returns:
        TC score in [0, 1]. Returns 0.0 if no reference columns.
    """
    # Find common sequence IDs
    pred_ids = {_normalize_id(r.id) for r in pred_msa}
    ref_ids = {_normalize_id(r.id) for r in ref_msa}
    common_ids = pred_ids & ref_ids

    if len(common_ids) < 2:
        return 0.0

    # Extract column signatures, filtering to common sequences only
    ref_columns = _extract_columns_for_subset(ref_msa, common_ids)
    if not ref_columns:
        return 0.0

    pred_columns = _extract_columns_for_subset(pred_msa, common_ids)
    correct = len(ref_columns & pred_columns)

    return correct / len(ref_columns)


def _extract_columns_for_subset(
    msa: MultipleSeqAlignment,
    subset_ids: set,
) -> Set[FrozenSet[Tuple[str, int]]]:
    """Extract column signatures considering only a subset of sequences.

    Args:
        msa: A MultipleSeqAlignment object.
        subset_ids: Set of normalized sequence IDs to include.

    Returns:
        Set of frozensets representing column signatures.
    """
    n_seqs = len(msa)
    alignment_len = msa.get_alignment_length()

    # Build ID and sequence lists, track which are in subset
    ids = [_normalize_id(msa[i].id) for i in range(n_seqs)]
    seqs = [_standardize_gaps(str(msa[i].seq)) for i in range(n_seqs)]
    in_subset = [ids[i] in subset_ids for i in range(n_seqs)]

    # Track residue index per sequence
    residue_indices = [0] * n_seqs

    columns = set()
    for col in range(alignment_len):
        signature = []
        for s in range(n_seqs):
            if col < len(seqs[s]) and seqs[s][col] not in GAP_CHARS:
                if in_subset[s]:
                    signature.append((ids[s], residue_indices[s]))
                residue_indices[s] += 1
            # gap: don't increment residue index

        if len(signature) >= 2:
            columns.add(frozenset(signature))

    return columns
