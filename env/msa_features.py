"""State feature extraction for MSA test cases (N-sequence inputs).

20-dimensional feature vector capturing MSA properties that help the agent
select appropriate MAFFT parameters.

Adapted from pilot's msa_state_features.py with inlined helper functions.
"""

import random
from collections import Counter
from typing import List

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
    if not seq:
        return 0.0
    count = sum(1 for c in seq.upper() if c in aa_set)
    return count / len(seq)


def _sequence_entropy(seq: str) -> float:
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
    return entropy / np.log2(20)


def _low_complexity_fraction(seq: str, min_run: int = 4) -> float:
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


def extract_msa_features(sequences: List[str]) -> np.ndarray:
    """Extract 20 features from a list of raw (ungapped) sequences.

    Features:
        0-2:   Size & scale (num_seqs, avg_len, total_residues)
        3-6:   Length heterogeneity (CV, min/max ratio, range, median ratio)
        7-11:  Pairwise similarity (mean/std/min/max Jaccard, twilight frac)
        12-17: Composition (hydro, charged, polar, small, aromatic, pro+gly)
        18-19: Complexity (entropy, low-complexity frac)

    Returns:
        numpy array of shape (20,) with float32 features.
    """
    n = len(sequences)
    lengths = [len(s) for s in sequences]

    # Size & scale
    num_sequences_norm = min(n, 150) / 150.0
    avg_len = np.mean(lengths) if lengths else 0.0
    avg_length_norm = min(avg_len / 2000.0, 1.0)
    total_residues_norm = min(sum(lengths) / 100000.0, 1.0)

    # Length heterogeneity
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    if len(lengths) > 1 and np.mean(lengths) > 0:
        length_cv = min(float(np.std(lengths) / np.mean(lengths)), 2.0)
    else:
        length_cv = 0.0
    min_max_length_ratio = min_len / max_len if max_len > 0 else 0.0
    length_range_norm = (max_len - min_len) / max_len if max_len > 0 else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0
    median_length_ratio = median_len / max_len if max_len > 0 else 0.0

    # Pairwise similarity (sampled, max 50 pairs)
    if n >= 2:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(all_pairs) > 50:
            sampled_pairs = random.sample(all_pairs, 50)
        else:
            sampled_pairs = all_pairs
        jaccards = [_kmer_jaccard(sequences[i].upper(), sequences[j].upper(), k=3)
                     for i, j in sampled_pairs]
        avg_pairwise_identity = float(np.mean(jaccards))
        pairwise_identity_std = float(np.std(jaccards))
        min_pairwise_identity = float(np.min(jaccards))
        max_pairwise_identity = float(np.max(jaccards))
        identity_twilight_frac = float(np.mean([1.0 if j < 0.3 else 0.0 for j in jaccards]))
    else:
        avg_pairwise_identity = 0.0
        pairwise_identity_std = 0.0
        min_pairwise_identity = 0.0
        max_pairwise_identity = 0.0
        identity_twilight_frac = 0.0

    # Composition
    avg_hydrophobic = float(np.mean([_aa_fraction(s, HYDROPHOBIC) for s in sequences]))
    avg_charged = float(np.mean([_aa_fraction(s, CHARGED) for s in sequences]))
    avg_polar = float(np.mean([_aa_fraction(s, POLAR) for s in sequences]))
    avg_small = float(np.mean([_aa_fraction(s, SMALL) for s in sequences]))
    avg_aromatic = float(np.mean([_aa_fraction(s, AROMATIC) for s in sequences]))
    avg_pg = float(np.mean([_aa_fraction(s, PROLINE_GLYCINE) for s in sequences]))

    # Complexity
    avg_entropy = float(np.mean([_sequence_entropy(s) for s in sequences]))
    avg_lc = float(np.mean([_low_complexity_fraction(s) for s in sequences]))

    return np.array([
        num_sequences_norm, avg_length_norm, total_residues_norm,
        length_cv, min_max_length_ratio, length_range_norm, median_length_ratio,
        avg_pairwise_identity, pairwise_identity_std, min_pairwise_identity,
        max_pairwise_identity, identity_twilight_frac,
        avg_hydrophobic, avg_charged, avg_polar, avg_small, avg_aromatic, avg_pg,
        avg_entropy, avg_lc,
    ], dtype=np.float32)
