"""PairwiseDataset class for sampling training/evaluation pairs."""

import random
from pathlib import Path
from typing import Dict, List, Optional

from data.balibase_parser import PairwiseReference, load_all_references


class PairwiseDataset:
    """Dataset of pairwise reference alignments from BAliBASE.

    Supports random sampling (for training) and deterministic iteration (for eval).
    """

    def __init__(self, references: List[PairwiseReference], max_seq_length: int = 2000):
        self.max_seq_length = max_seq_length

        # Filter by max sequence length
        self.references = [
            ref for ref in references
            if len(ref.seq1_raw) <= max_seq_length and len(ref.seq2_raw) <= max_seq_length
        ]

        filtered = len(references) - len(self.references)
        if filtered > 0:
            print(f"  Filtered {filtered} pairs exceeding max_seq_length={max_seq_length}")

    def __len__(self) -> int:
        return len(self.references)

    def sample(self, n: int = 1) -> List[PairwiseReference]:
        """Sample n random pairs (with replacement)."""
        return random.choices(self.references, k=n)

    def sample_one(self) -> PairwiseReference:
        """Sample a single random pair."""
        return random.choice(self.references)

    def __iter__(self):
        """Deterministic iteration over all pairs (for evaluation)."""
        return iter(self.references)

    def __getitem__(self, idx: int) -> PairwiseReference:
        return self.references[idx]

    def get_by_ref_set(self) -> Dict[str, List[PairwiseReference]]:
        """Group references by reference set."""
        by_set: Dict[str, List[PairwiseReference]] = {}
        for ref in self.references:
            by_set.setdefault(ref.ref_set, []).append(ref)
        return by_set


def build_datasets(
    data_dir: Path,
    train_ref_sets: List[str],
    eval_ref_sets: List[str],
    max_seq_length: int = 2000,
) -> tuple:
    """Build train and eval datasets from BAliBASE.

    Returns (train_dataset, eval_dataset, all_refs_dict).
    """
    all_sets = list(set(train_ref_sets + eval_ref_sets))
    print("Loading BAliBASE references...")
    all_refs = load_all_references(data_dir, all_sets)

    train_refs = []
    for rs in train_ref_sets:
        train_refs.extend(all_refs.get(rs, []))

    eval_refs = []
    for rs in eval_ref_sets:
        eval_refs.extend(all_refs.get(rs, []))

    train_dataset = PairwiseDataset(train_refs, max_seq_length)
    eval_dataset = PairwiseDataset(eval_refs, max_seq_length)

    print(f"Train: {len(train_dataset)} pairs, Eval: {len(eval_dataset)} pairs")
    return train_dataset, eval_dataset, all_refs
