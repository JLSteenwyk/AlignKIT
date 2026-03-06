"""Diversity-preserving subsampling for MSA benchmark datasets.

Uses farthest-point sampling (maxmin diversity) in the 20-dimensional
state feature space to select a representative subset that maximizes
coverage of the sequence-property landscape.

Algorithm:
    1. Extract 20-dim feature vector for each MSA test case
    2. Z-score normalize features (so all dimensions contribute equally)
    3. Seed with the medoid (point closest to centroid)
    4. Iteratively select the point whose minimum distance to all
       already-selected points is largest
    5. Repeat until target count is reached

This guarantees that outliers (unusual MSAs) are picked early and the
selected set fills the feature space as evenly as possible.
"""

from typing import Callable, List

import numpy as np

from data.msa_dataset import MSATestCase


def farthest_point_subsample(
    cases: List[MSATestCase],
    feature_fn: Callable[[List[str]], np.ndarray],
    target_n: int,
) -> List[MSATestCase]:
    """Select target_n cases maximizing diversity via farthest-point sampling.

    Args:
        cases: Full list of MSA test cases to subsample from.
        feature_fn: Function mapping sequences -> feature vector (e.g.
            extract_msa_features).
        target_n: Number of cases to select.

    Returns:
        List of target_n MSATestCase objects with maximal diversity.
        If len(cases) <= target_n, returns all cases unchanged.
    """
    if len(cases) <= target_n:
        return list(cases)

    # Extract features for all cases
    features = np.array([feature_fn(case.sequences) for case in cases])
    n_cases, n_feat = features.shape

    # Z-score normalize so all feature dimensions contribute equally
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    features_norm = (features - mean) / std

    # Seed with medoid (point closest to centroid)
    centroid = features_norm.mean(axis=0)
    dists_to_centroid = np.linalg.norm(features_norm - centroid, axis=1)
    medoid_idx = int(np.argmin(dists_to_centroid))

    selected = [medoid_idx]
    selected_mask = np.zeros(n_cases, dtype=bool)
    selected_mask[medoid_idx] = True

    # min_dists[i] = min distance from case i to any selected case
    min_dists = np.linalg.norm(
        features_norm - features_norm[medoid_idx], axis=1
    )
    min_dists[selected_mask] = -np.inf

    for _ in range(target_n - 1):
        # Pick the point farthest from its nearest selected neighbor
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        selected_mask[next_idx] = True

        # Update min distances with the newly added point
        new_dists = np.linalg.norm(
            features_norm - features_norm[next_idx], axis=1
        )
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected_mask] = -np.inf

    return [cases[i] for i in selected]
