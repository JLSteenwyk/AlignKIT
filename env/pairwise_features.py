"""12-dimensional feature extraction for sequence pairs.

Features:
  Length (3):    ratio, normalized max, log product
  Similarity (2): k-mer Jaccard k=2 and k=3
  Composition (4): hydrophobic, charged, polar, Pro+Gly fractions
  Complexity (2): entropy, low-complexity fraction
  Divergence (1): Jensen-Shannon of AA frequencies
"""

import math
from collections import Counter
from typing import Tuple

import numpy as np

# Amino acid property groups
HYDROPHOBIC = set("AILMFWVP")
CHARGED = set("DEKRH")
POLAR = set("STNQYC")
PRO_GLY = set("PG")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _length_features(seq1: str, seq2: str) -> Tuple[float, float, float]:
    """Length-based features."""
    l1, l2 = len(seq1), len(seq2)
    if l1 == 0 or l2 == 0:
        return 0.0, 0.0, 0.0
    ratio = min(l1, l2) / max(l1, l2)
    norm_max = max(l1, l2) / 500.0  # normalized by max_seq_length
    log_prod = math.log(l1 * l2 + 1) / 12.0  # normalize ~log(500*500)
    return ratio, norm_max, log_prod


def _kmer_jaccard(seq1: str, seq2: str, k: int) -> float:
    """Jaccard similarity of k-mer sets."""
    if len(seq1) < k or len(seq2) < k:
        return 0.0
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    if not kmers1 and not kmers2:
        return 0.0
    return len(kmers1 & kmers2) / len(kmers1 | kmers2)


def _group_fraction(seq: str, group: set) -> float:
    """Fraction of residues belonging to a group."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in group) / len(seq)


def _aa_entropy(seq: str) -> float:
    """Shannon entropy of amino acid frequencies, normalized to [0,1]."""
    if not seq:
        return 0.0
    counts = Counter(c for c in seq if c in STANDARD_AA)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    max_entropy = math.log(min(total, 20))
    if max_entropy == 0:
        return 0.0
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values() if n > 0)
    return entropy / max_entropy


def _low_complexity_fraction(seq: str, window: int = 12, threshold: float = 1.5) -> float:
    """Fraction of positions in low-complexity windows."""
    if len(seq) < window:
        return 0.0
    low_count = 0
    for i in range(len(seq) - window + 1):
        w = seq[i:i+window]
        counts = Counter(w)
        entropy = -sum((n / window) * math.log(n / window) for n in counts.values() if n > 0)
        if entropy < threshold:
            low_count += 1
    return low_count / (len(seq) - window + 1)


def _aa_frequency(seq: str) -> np.ndarray:
    """20-dim amino acid frequency vector."""
    freq = np.zeros(20, dtype=np.float64)
    total = 0
    for c in seq:
        if c in STANDARD_AA:
            idx = "ACDEFGHIKLMNPQRSTVWY".index(c)
            freq[idx] += 1
            total += 1
    if total > 0:
        freq /= total
    return freq


def _jensen_shannon(seq1: str, seq2: str) -> float:
    """Jensen-Shannon divergence of amino acid frequencies, normalized."""
    p = _aa_frequency(seq1)
    q = _aa_frequency(seq2)
    m = 0.5 * (p + q)

    def _kl(a, b):
        kl = 0.0
        for i in range(len(a)):
            if a[i] > 0 and b[i] > 0:
                kl += a[i] * math.log(a[i] / b[i])
        return kl

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Normalize by log(2) so JSD is in [0, 1]
    return jsd / math.log(2) if jsd > 0 else 0.0


def extract_pairwise_features(seq1: str, seq2: str) -> np.ndarray:
    """Extract 12-dimensional feature vector from a sequence pair.

    Args:
        seq1: Raw amino acid sequence 1 (uppercase, no gaps).
        seq2: Raw amino acid sequence 2 (uppercase, no gaps).

    Returns:
        numpy array of float32, shape (12,).
    """
    # Length features (3)
    ratio, norm_max, log_prod = _length_features(seq1, seq2)

    # Similarity features (2)
    jac2 = _kmer_jaccard(seq1, seq2, 2)
    jac3 = _kmer_jaccard(seq1, seq2, 3)

    # Composition features (4) — averaged over both sequences
    concat = seq1 + seq2
    hydro = _group_fraction(concat, HYDROPHOBIC)
    charged = _group_fraction(concat, CHARGED)
    polar = _group_fraction(concat, POLAR)
    pro_gly = _group_fraction(concat, PRO_GLY)

    # Complexity features (2) — averaged
    ent = 0.5 * (_aa_entropy(seq1) + _aa_entropy(seq2))
    lc = 0.5 * (_low_complexity_fraction(seq1) + _low_complexity_fraction(seq2))

    # Divergence (1)
    jsd = _jensen_shannon(seq1, seq2)

    features = np.array([
        ratio, norm_max, log_prod,
        jac2, jac3,
        hydro, charged, polar, pro_gly,
        ent, lc,
        jsd,
    ], dtype=np.float32)

    return features
