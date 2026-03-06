"""Pairwise dataset: extract sequence pairs from BAliBASE MSAs.

Uses balibase_parser.extract_pairwise_references() to get all pairs,
then precomputes features and integer-encoded sequences.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.balibase_parser import PairwiseReference, load_all_references
from env.needleman_wunsch import encode_sequence
from env.pairwise_features import extract_pairwise_features


@dataclass
class PairwiseExample:
    """A precomputed pairwise example ready for training/eval."""
    seq1_raw: str
    seq2_raw: str
    seq1_idx: np.ndarray   # integer-encoded, shape (L1,)
    seq2_idx: np.ndarray   # integer-encoded, shape (L2,)
    features: np.ndarray   # shape (12,)
    ref_seq1_aligned: str  # reference alignment
    ref_seq2_aligned: str
    ref_set: str
    pair_id: str           # e.g. "BB11001:seq1:seq2"


class PairwiseDataset:
    """Dataset of pairwise alignment examples.

    Supports random sampling (training) and deterministic iteration (eval).
    """

    def __init__(self, examples: List[PairwiseExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def sample_one(self) -> PairwiseExample:
        return random.choice(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, idx: int) -> PairwiseExample:
        return self.examples[idx]

    def get_by_ref_set(self) -> Dict[str, List[PairwiseExample]]:
        by_set: Dict[str, List[PairwiseExample]] = {}
        for ex in self.examples:
            by_set.setdefault(ex.ref_set, []).append(ex)
        return by_set


def _build_examples(
    refs: Dict[str, List[PairwiseReference]],
    max_seq_length: int,
) -> List[PairwiseExample]:
    """Convert PairwiseReferences into precomputed PairwiseExamples."""
    examples = []
    skipped = 0

    for ref_set, pairs in refs.items():
        for pair in pairs:
            # Skip pairs exceeding max length
            if len(pair.seq1_raw) > max_seq_length or len(pair.seq2_raw) > max_seq_length:
                skipped += 1
                continue

            seq1_idx = encode_sequence(pair.seq1_raw)
            seq2_idx = encode_sequence(pair.seq2_raw)
            features = extract_pairwise_features(pair.seq1_raw, pair.seq2_raw)

            pair_id = f"{pair.source_file}:{pair.seq1_id}:{pair.seq2_id}"

            examples.append(PairwiseExample(
                seq1_raw=pair.seq1_raw,
                seq2_raw=pair.seq2_raw,
                seq1_idx=seq1_idx,
                seq2_idx=seq2_idx,
                features=features,
                ref_seq1_aligned=pair.seq1_aligned,
                ref_seq2_aligned=pair.seq2_aligned,
                ref_set=pair.ref_set,
                pair_id=pair_id,
            ))

    if skipped > 0:
        print(f"  Skipped {skipped} pairs exceeding max_seq_length={max_seq_length}")

    return examples


def build_pairwise_datasets(
    data_dir: Path,
    train_ref_sets: List[str],
    eval_ref_sets: List[str],
    max_seq_length: int = 500,
) -> tuple:
    """Build train and eval pairwise datasets from BAliBASE.

    Train/eval split is by reference set (not random), so we test
    generalization to different difficulty levels.

    Returns:
        (train_dataset, eval_dataset)
    """
    print("Loading BAliBASE pairwise references...")

    all_ref_sets = list(set(train_ref_sets + eval_ref_sets))
    all_refs = load_all_references(data_dir, all_ref_sets)

    # Split by ref_set
    train_refs = {rs: all_refs.get(rs, []) for rs in train_ref_sets}
    eval_refs = {rs: all_refs.get(rs, []) for rs in eval_ref_sets}

    print("Building train examples...")
    train_examples = _build_examples(train_refs, max_seq_length)
    print(f"  Train: {len(train_examples)} pairs")

    print("Building eval examples...")
    eval_examples = _build_examples(eval_refs, max_seq_length)
    print(f"  Eval: {len(eval_examples)} pairs")

    return PairwiseDataset(train_examples), PairwiseDataset(eval_examples)
