"""State feature extraction for MSA test cases (N-sequence inputs).

20-dimensional feature vector capturing MSA properties that help the agent
select appropriate tool and parameter configurations.
"""

import random
from typing import List

import numpy as np

from env.state_features import (
    _kmer_jaccard, _aa_fraction, _sequence_entropy, _low_complexity_fraction,
    HYDROPHOBIC, CHARGED, POLAR, SMALL, AROMATIC, PROLINE_GLYCINE,
)


def extract_msa_features(sequences: List[str]) -> np.ndarray:
    """Extract 20 features from a list of raw (ungapped) sequences.

    Features:
        --- Size & scale ---
        0.  num_sequences_norm:        min(N, 150) / 150
        1.  avg_length_norm:           mean(lengths) / 2000
        2.  total_residues_norm:       min(sum(lengths), 100000) / 100000

        --- Length heterogeneity ---
        3.  length_cv:                 std/mean of lengths (capped at 2)
        4.  min_max_length_ratio:      min(lengths) / max(lengths)
        5.  length_range_norm:         (max - min) / max lengths
        6.  median_length_ratio:       median(lengths) / max(lengths)

        --- Pairwise similarity ---
        7.  avg_pairwise_identity:     mean k-mer Jaccard (sampled, max 50 pairs)
        8.  pairwise_identity_std:     std of k-mer Jaccard
        9.  min_pairwise_identity:     min k-mer Jaccard — hardest pair
        10. max_pairwise_identity:     max k-mer Jaccard — closest pair
        11. identity_twilight_frac:    fraction of pairs with Jaccard < 0.3

        --- Sequence composition ---
        12. avg_hydrophobic_fraction:  mean AILMFWVP fraction
        13. avg_charged_fraction:      mean DEKRH fraction
        14. avg_polar_fraction:        mean STNQ fraction
        15. avg_small_fraction:        mean GASTCV fraction
        16. avg_aromatic_fraction:     mean FWY fraction
        17. avg_proline_glycine_frac:  mean PG fraction

        --- Sequence complexity ---
        18. avg_sequence_entropy:      mean normalized AA entropy
        19. avg_low_complexity_frac:   mean fraction of residues in homopolymer runs >= 4

    Args:
        sequences: List of raw sequences (no gaps).

    Returns:
        numpy array of shape (20,) with float32 features.
    """
    n = len(sequences)
    lengths = [len(s) for s in sequences]

    # --- Size & scale ---
    # 0. Normalized number of sequences
    num_sequences_norm = min(n, 150) / 150.0

    # 1. Normalized average length
    avg_len = np.mean(lengths) if lengths else 0.0
    avg_length_norm = min(avg_len / 2000.0, 1.0)

    # 2. Total residues normalized
    total_residues = sum(lengths)
    total_residues_norm = min(total_residues / 100000.0, 1.0)

    # --- Length heterogeneity ---
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    # 3. Length coefficient of variation
    if len(lengths) > 1 and np.mean(lengths) > 0:
        length_cv = min(float(np.std(lengths) / np.mean(lengths)), 2.0)
    else:
        length_cv = 0.0

    # 4. Min/max length ratio
    min_max_length_ratio = min_len / max_len if max_len > 0 else 0.0

    # 5. Length range normalized
    length_range_norm = (max_len - min_len) / max_len if max_len > 0 else 0.0

    # 6. Median/max length ratio
    median_len = float(np.median(lengths)) if lengths else 0.0
    median_length_ratio = median_len / max_len if max_len > 0 else 0.0

    # --- Pairwise similarity ---
    if n >= 2:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(all_pairs) > 50:
            sampled_pairs = random.sample(all_pairs, 50)
        else:
            sampled_pairs = all_pairs

        jaccards = []
        for i, j in sampled_pairs:
            jaccards.append(_kmer_jaccard(sequences[i].upper(), sequences[j].upper(), k=3))

        # 7. Mean pairwise identity
        avg_pairwise_identity = float(np.mean(jaccards))
        # 8. Std pairwise identity
        pairwise_identity_std = float(np.std(jaccards))
        # 9. Min pairwise identity
        min_pairwise_identity = float(np.min(jaccards))
        # 10. Max pairwise identity
        max_pairwise_identity = float(np.max(jaccards))
        # 11. Twilight zone fraction (Jaccard < 0.3)
        identity_twilight_frac = float(np.mean([1.0 if j < 0.3 else 0.0 for j in jaccards]))
    else:
        avg_pairwise_identity = 0.0
        pairwise_identity_std = 0.0
        min_pairwise_identity = 0.0
        max_pairwise_identity = 0.0
        identity_twilight_frac = 0.0

    # --- Sequence composition ---
    # 12. Hydrophobic
    hydro_fracs = [_aa_fraction(s, HYDROPHOBIC) for s in sequences]
    avg_hydrophobic_fraction = float(np.mean(hydro_fracs)) if hydro_fracs else 0.0

    # 13. Charged
    charged_fracs = [_aa_fraction(s, CHARGED) for s in sequences]
    avg_charged_fraction = float(np.mean(charged_fracs)) if charged_fracs else 0.0

    # 14. Polar
    polar_fracs = [_aa_fraction(s, POLAR) for s in sequences]
    avg_polar_fraction = float(np.mean(polar_fracs)) if polar_fracs else 0.0

    # 15. Small
    small_fracs = [_aa_fraction(s, SMALL) for s in sequences]
    avg_small_fraction = float(np.mean(small_fracs)) if small_fracs else 0.0

    # 16. Aromatic
    aromatic_fracs = [_aa_fraction(s, AROMATIC) for s in sequences]
    avg_aromatic_fraction = float(np.mean(aromatic_fracs)) if aromatic_fracs else 0.0

    # 17. Proline + Glycine
    pg_fracs = [_aa_fraction(s, PROLINE_GLYCINE) for s in sequences]
    avg_proline_glycine_frac = float(np.mean(pg_fracs)) if pg_fracs else 0.0

    # --- Sequence complexity ---
    # 18. Average normalized AA entropy
    entropies = [_sequence_entropy(s) for s in sequences]
    avg_sequence_entropy = float(np.mean(entropies)) if entropies else 0.0

    # 19. Average low-complexity fraction
    lc_fracs = [_low_complexity_fraction(s) for s in sequences]
    avg_low_complexity_frac = float(np.mean(lc_fracs)) if lc_fracs else 0.0

    return np.array(
        [
            num_sequences_norm,         # 0
            avg_length_norm,            # 1
            total_residues_norm,        # 2
            length_cv,                  # 3
            min_max_length_ratio,       # 4
            length_range_norm,          # 5
            median_length_ratio,        # 6
            avg_pairwise_identity,      # 7
            pairwise_identity_std,      # 8
            min_pairwise_identity,      # 9
            max_pairwise_identity,      # 10
            identity_twilight_frac,     # 11
            avg_hydrophobic_fraction,   # 12
            avg_charged_fraction,       # 13
            avg_polar_fraction,         # 14
            avg_small_fraction,         # 15
            avg_aromatic_fraction,      # 16
            avg_proline_glycine_frac,   # 17
            avg_sequence_entropy,       # 18
            avg_low_complexity_frac,    # 19
        ],
        dtype=np.float32,
    )
