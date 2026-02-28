"""State feature extraction for sequence pairs.

Features capture properties of the input sequence pair that may help
the agent select appropriate alignment parameters.
"""

from collections import Counter

import numpy as np

# Standard amino acid property sets
HYDROPHOBIC = frozenset("AILMFWVP")
CHARGED = frozenset("DEKRH")
POLAR = frozenset("STNQ")
SMALL = frozenset("GASTCV")
AROMATIC = frozenset("FWY")
PROLINE_GLYCINE = frozenset("PG")
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _kmer_jaccard(seq1: str, seq2: str, k: int = 3) -> float:
    """Compute Jaccard similarity of k-mer sets between two sequences."""
    if len(seq1) < k or len(seq2) < k:
        return 0.0

    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))

    if not kmers1 and not kmers2:
        return 0.0

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union if union > 0 else 0.0


def _aa_fraction(seq: str, aa_set: frozenset) -> float:
    """Fraction of residues in seq belonging to aa_set."""
    if not seq:
        return 0.0
    count = sum(1 for c in seq.upper() if c in aa_set)
    return count / len(seq)


def _sequence_entropy(seq: str) -> float:
    """Average per-residue Shannon entropy of amino acid composition."""
    seq_upper = seq.upper()
    n = len(seq_upper)
    if n == 0:
        return 0.0
    counts = Counter(c for c in seq_upper if c in STANDARD_AA)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * np.log2(p)
    # Normalize by max possible entropy (log2(20))
    return entropy / np.log2(20)


def _low_complexity_fraction(seq: str, min_run: int = 4) -> float:
    """Fraction of residues in homopolymer runs of length >= min_run."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    n = len(seq_upper)
    in_run = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and seq_upper[j] == seq_upper[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            in_run += run_len
        i = j
    return in_run / n


def extract_features(seq1: str, seq2: str) -> np.ndarray:
    """Extract 5 features from a pair of raw (ungapped) sequences.

    Features:
        0. length_ratio: min(len1, len2) / max(len1, len2) — length similarity
        1. normalized_max_length: max(len1, len2) / 2000 — absolute scale (capped at 1)
        2. kmer_jaccard_identity: Jaccard similarity of 3-mer sets — sequence similarity proxy
        3. hydrophobic_fraction: avg fraction of hydrophobic residues — composition signal
        4. charged_fraction: avg fraction of charged residues — composition signal

    Args:
        seq1: Raw sequence 1 (no gaps).
        seq2: Raw sequence 2 (no gaps).

    Returns:
        numpy array of shape (5,) with float32 features.
    """
    len1, len2 = len(seq1), len(seq2)

    # Length ratio
    if max(len1, len2) > 0:
        length_ratio = min(len1, len2) / max(len1, len2)
    else:
        length_ratio = 0.0

    # Normalized max length (capped at 1.0)
    normalized_max_length = min(max(len1, len2) / 2000.0, 1.0)

    # K-mer Jaccard similarity
    kmer_identity = _kmer_jaccard(seq1.upper(), seq2.upper(), k=3)

    # Average hydrophobic fraction
    hydro1 = _aa_fraction(seq1, HYDROPHOBIC)
    hydro2 = _aa_fraction(seq2, HYDROPHOBIC)
    hydrophobic_fraction = (hydro1 + hydro2) / 2.0

    # Average charged fraction
    charged1 = _aa_fraction(seq1, CHARGED)
    charged2 = _aa_fraction(seq2, CHARGED)
    charged_fraction = (charged1 + charged2) / 2.0

    return np.array(
        [length_ratio, normalized_max_length, kmer_identity,
         hydrophobic_fraction, charged_fraction],
        dtype=np.float32,
    )
