"""Needleman-Wunsch with position-dependent affine gap penalties (Gotoh).

Standard 3-matrix recurrence:
  D[i,j]: best score ending in match/mismatch at (i,j)
  P[i,j]: best score ending in gap in seq2 (consuming seq1[i])
  Q[i,j]: best score ending in gap in seq1 (consuming seq2[j])

Region mapping: each sequence is divided into K equal segments.
Position p in sequence of length L maps to region min(p * K / L, K-1).

Performance: D and P are vectorized per row with numpy. Q requires a
sequential scan (left-to-right dependency) done with Python lists to
avoid numpy scalar indexing overhead.
"""

import numpy as np

# Standard BLOSUM62 matrix
# Order: A R N D C Q E G H I L K M F P S T W Y V
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
], dtype=np.float64)

# Score for unknown / non-standard amino acids
UNKNOWN_SCORE = -1.0


def encode_sequence(seq: str) -> np.ndarray:
    """Encode amino acid sequence as integer indices into AA_ORDER.

    Unknown residues get index -1.

    Args:
        seq: Amino acid sequence string (uppercase).

    Returns:
        numpy array of int32 indices, shape (len(seq),).
    """
    return np.array([AA_TO_IDX.get(c, -1) for c in seq.upper()], dtype=np.int32)


def region_map(seq_len: int, num_regions: int) -> np.ndarray:
    """Map each position in a sequence to its region index.

    Position p maps to min(p * K / L, K-1).

    Args:
        seq_len: Length of the sequence.
        num_regions: Number of regions (K).

    Returns:
        numpy array of int32, shape (seq_len,), values in [0, K-1].
    """
    if seq_len == 0:
        return np.array([], dtype=np.int32)
    positions = np.arange(seq_len, dtype=np.int32)
    regions = np.minimum(positions * num_regions // seq_len, num_regions - 1)
    return regions.astype(np.int32)


def needleman_wunsch(
    seq1_idx: np.ndarray,
    seq2_idx: np.ndarray,
    gap_open: np.ndarray,
    gap_extend: np.ndarray,
    num_regions: int,
) -> tuple:
    """Needleman-Wunsch alignment with position-dependent affine gap penalties.

    Uses Gotoh's 3-matrix formulation. D and P are vectorized per row;
    Q uses a sequential scan with Python lists for speed.

    Args:
        seq1_idx: Integer-encoded sequence 1, shape (L1,).
        seq2_idx: Integer-encoded sequence 2, shape (L2,).
        gap_open: Gap opening penalties per region, shape (K,). Positive values.
        gap_extend: Gap extension penalties per region, shape (K,). Positive values.
        num_regions: Number of regions K.

    Returns:
        (aligned_seq1, aligned_seq2, score) where aligned sequences are
        strings using AA_ORDER characters and '-' for gaps.
    """
    L1 = len(seq1_idx)
    L2 = len(seq2_idx)

    reg1 = region_map(L1, num_regions)
    reg2 = region_map(L2, num_regions)

    NEG_INF = -1e9

    # --- Precompute substitution score matrix (L1 x L2) ---
    safe1 = np.maximum(seq1_idx, 0)
    safe2 = np.maximum(seq2_idx, 0)
    sub_matrix = BLOSUM62[safe1[:, None], safe2[None, :]].copy()
    unk1 = seq1_idx < 0
    unk2 = seq2_idx < 0
    if unk1.any() or unk2.any():
        sub_matrix[unk1[:, None] | unk2[None, :]] = UNKNOWN_SCORE

    # --- Per-position gap penalties (numpy arrays) ---
    go1 = gap_open[reg1]   # (L1,)
    ge1 = gap_extend[reg1]
    go2 = gap_open[reg2]   # (L2,)
    ge2 = gap_extend[reg2]

    # Python lists for the Q scan inner loop
    go2_list = go2.tolist()
    ge2_list = ge2.tolist()

    # --- DP matrices (1-indexed) ---
    D = np.full((L1 + 1, L2 + 1), NEG_INF, dtype=np.float64)
    P = np.full((L1 + 1, L2 + 1), NEG_INF, dtype=np.float64)
    Q = np.full((L1 + 1, L2 + 1), NEG_INF, dtype=np.float64)

    D[0, 0] = 0.0

    # Vectorized first-row init: gaps in seq1 (consuming seq2)
    j_arr = np.arange(1, L2 + 1)
    Q[0, 1:] = -go2 - ge2 * (j_arr - 1)
    D[0, 1:] = Q[0, 1:]

    # Vectorized first-column init: gaps in seq2 (consuming seq1)
    i_arr = np.arange(1, L1 + 1)
    P[1:, 0] = -go1 - ge1 * (i_arr - 1)
    D[1:, 0] = P[1:, 0]

    # --- Row-by-row DP fill ---
    for i in range(1, L1 + 1):
        # D[i, 1:] = sub_matrix[i-1, :] + max(D[i-1, 0:L2], P[i-1, 0:L2], Q[i-1, 0:L2])
        prev_best = np.maximum(np.maximum(D[i-1, :L2], P[i-1, :L2]), Q[i-1, :L2])
        D[i, 1:] = sub_matrix[i - 1] + prev_best

        # P[i, 1:] = max(D[i-1, 1:] - go1[i-1], P[i-1, 1:] - ge1[i-1])
        go_i = go1[i - 1]
        ge_i = ge1[i - 1]
        P[i, 1:] = np.maximum(D[i-1, 1:] - go_i, P[i-1, 1:] - ge_i)

        # Q[i, :] sequential scan using Python lists (avoids numpy scalar overhead)
        d_list = D[i].tolist()
        q_prev = NEG_INF
        q_new = [NEG_INF] * (L2 + 1)
        for j in range(1, L2 + 1):
            o = d_list[j - 1] - go2_list[j - 1]
            e = q_prev - ge2_list[j - 1]
            q_prev = o if o >= e else e
            q_new[j] = q_prev
        Q[i, :] = q_new

    # --- Final score ---
    final_score = max(float(D[L1, L2]), float(P[L1, L2]), float(Q[L1, L2]))

    # --- Traceback ---
    chars1 = [AA_ORDER[idx] if idx >= 0 else 'X' for idx in seq1_idx]
    chars2 = [AA_ORDER[idx] if idx >= 0 else 'X' for idx in seq2_idx]

    aligned1 = []
    aligned2 = []
    i, j = L1, L2

    end_scores = [D[L1, L2], P[L1, L2], Q[L1, L2]]
    state = int(np.argmax(end_scores))  # 0=D, 1=P, 2=Q

    while i > 0 or j > 0:
        if state == 0:  # D state: came from diagonal
            if i > 0 and j > 0:
                aligned1.append(chars1[i - 1])
                aligned2.append(chars2[j - 1])

                val = D[i, j] - sub_matrix[i - 1, j - 1]

                if abs(val - D[i-1, j-1]) < 1e-6:
                    state = 0
                elif abs(val - P[i-1, j-1]) < 1e-6:
                    state = 1
                else:
                    state = 2
                i -= 1
                j -= 1
            elif i > 0:
                aligned1.append(chars1[i - 1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(chars2[j - 1])
                j -= 1

        elif state == 1:  # P state: gap in seq2, consuming seq1[i]
            if i > 0:
                aligned1.append(chars1[i - 1])
                aligned2.append('-')

                if abs(P[i, j] - (D[i-1, j] - go1[i - 1])) < 1e-6:
                    state = 0
                else:
                    state = 1
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(chars2[j - 1])
                j -= 1
                state = 0

        elif state == 2:  # Q state: gap in seq1, consuming seq2[j]
            if j > 0:
                aligned1.append('-')
                aligned2.append(chars2[j - 1])

                if abs(Q[i, j] - (D[i, j-1] - go2[j - 1])) < 1e-6:
                    state = 0
                else:
                    state = 2
                j -= 1
            else:
                aligned1.append(chars1[i - 1])
                aligned2.append('-')
                i -= 1
                state = 0

    return "".join(reversed(aligned1)), "".join(reversed(aligned2)), final_score


def nw_align(
    seq1: str,
    seq2: str,
    gap_open: np.ndarray,
    gap_extend: np.ndarray,
    num_regions: int,
) -> tuple:
    """Convenience wrapper: takes string sequences, returns aligned strings + score.

    Args:
        seq1: Raw amino acid sequence 1 (uppercase, no gaps).
        seq2: Raw amino acid sequence 2 (uppercase, no gaps).
        gap_open: Gap opening penalties per region, shape (K,).
        gap_extend: Gap extension penalties per region, shape (K,).
        num_regions: Number of regions K.

    Returns:
        (aligned_seq1, aligned_seq2, score)
    """
    seq1_idx = encode_sequence(seq1)
    seq2_idx = encode_sequence(seq2)
    return needleman_wunsch(seq1_idx, seq2_idx, gap_open, gap_extend, num_regions)
