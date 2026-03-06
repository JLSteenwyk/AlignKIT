"""MSA-level dataset: load BAliBASE test cases as whole MSAs (not pairwise)."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment

from data.balibase_parser import parse_balibase_alignment, _normalize_id


@dataclass
class MSATestCase:
    """A single BAliBASE MSA test case."""
    case_id: str                        # e.g. "BB11001"
    ref_set: str                        # e.g. "RV11"
    fasta_path: Path                    # input .tfa file
    msf_path: Path                      # reference .msf file
    num_sequences: int                  # number of sequences
    sequences: List[str]                # raw (ungapped) sequences
    seq_ids: List[str]                  # normalized sequence IDs
    ref_alignment: MultipleSeqAlignment # parsed reference MSA


class MSADataset:
    """Dataset of MSA test cases from BAliBASE.

    Supports random sampling (for training) and deterministic iteration (for eval).
    """

    def __init__(self, test_cases: List[MSATestCase]):
        self.test_cases = test_cases

    def __len__(self) -> int:
        return len(self.test_cases)

    def sample_one(self) -> MSATestCase:
        """Sample a single random test case."""
        return random.choice(self.test_cases)

    def sample(self, n: int = 1) -> List[MSATestCase]:
        """Sample n random test cases (with replacement)."""
        return random.choices(self.test_cases, k=n)

    def __iter__(self):
        """Deterministic iteration over all test cases."""
        return iter(self.test_cases)

    def __getitem__(self, idx: int) -> MSATestCase:
        return self.test_cases[idx]

    def get_by_ref_set(self) -> Dict[str, List[MSATestCase]]:
        """Group test cases by reference set."""
        by_set: Dict[str, List[MSATestCase]] = {}
        for tc in self.test_cases:
            by_set.setdefault(tc.ref_set, []).append(tc)
        return by_set


def load_msa_test_cases(data_dir: Path, ref_sets: List[str]) -> List[MSATestCase]:
    """Load MSA test cases from BAliBASE directory.

    For each reference set, finds BB[0-9]*.tfa input files and matches
    them to BB[0-9]*.msf reference alignments.

    Args:
        data_dir: Path to bb3_release directory.
        ref_sets: List of reference set names (e.g. ["RV11", "RV12"]).

    Returns:
        List of MSATestCase objects.
    """
    test_cases = []

    for ref_set in ref_sets:
        ref_dir = data_dir / ref_set
        if not ref_dir.exists():
            print(f"  Warning: {ref_dir} not found, skipping")
            continue

        # Find input FASTA files (BB*.tfa, not BBS*)
        tfa_files = sorted(ref_dir.glob("BB[0-9]*.tfa"))

        loaded = 0
        for tfa_path in tfa_files:
            case_id = tfa_path.stem  # e.g. "BB11001"

            # Find matching MSF reference
            msf_path = ref_dir / f"{case_id}.msf"
            if not msf_path.exists():
                continue

            # Parse input sequences
            try:
                records = list(SeqIO.parse(tfa_path, "fasta"))
            except Exception:
                continue

            if len(records) < 2:
                continue

            sequences = [str(rec.seq).replace("-", "").replace(".", "") for rec in records]
            seq_ids = [_normalize_id(rec.id) for rec in records]

            # Parse reference alignment
            ref_alignment = parse_balibase_alignment(msf_path)
            if ref_alignment is None:
                continue

            test_cases.append(MSATestCase(
                case_id=case_id,
                ref_set=ref_set,
                fasta_path=tfa_path,
                msf_path=msf_path,
                num_sequences=len(records),
                sequences=sequences,
                seq_ids=seq_ids,
                ref_alignment=ref_alignment,
            ))
            loaded += 1

        print(f"  {ref_set}: loaded {loaded} MSA test cases from {len(tfa_files)} .tfa files")

    return test_cases


def build_msa_datasets(
    data_dir: Path,
    all_ref_sets: List[str],
    split_ratio: float = 0.8,
    split_seed: int = 42,
    benchmark_dir: Optional[Path] = None,
    subsample_benchmarks_to: Optional[int] = None,
) -> tuple:
    """Build train and eval MSA datasets with stratified random split.

    Loads BAliBASE ref_sets and optionally OXBench/SABRE/HOMSTRAD benchmarks,
    then splits 80/20 within each dataset/ref_set so that every source is
    represented in both train and eval.

    If subsample_benchmarks_to is set, each external benchmark's training
    portion is subsampled to that many cases using farthest-point sampling
    in the 20-dim state feature space.  This equalizes database representation
    while preserving the diversity within each benchmark.  The eval split
    is left untouched so evaluation remains comprehensive.

    Returns (train_dataset, eval_dataset).
    """
    print("Loading BAliBASE MSA test cases...")
    all_cases = load_msa_test_cases(data_dir, all_ref_sets)

    # Optionally load external benchmarks
    if benchmark_dir is not None and benchmark_dir.exists():
        from data.benchmark_loader import load_all_benchmarks
        print("Loading external benchmark datasets...")
        benchmark_cases = load_all_benchmarks(benchmark_dir)
        all_cases.extend(benchmark_cases)

    # Group by ref_set for stratified split
    by_set: Dict[str, List[MSATestCase]] = {}
    for tc in all_cases:
        by_set.setdefault(tc.ref_set, []).append(tc)

    rng = random.Random(split_seed)
    train_cases = []
    eval_cases = []

    for ref_set in sorted(by_set.keys()):
        cases = by_set[ref_set]
        rng.shuffle(cases)
        n_train = max(1, int(len(cases) * split_ratio))
        train_cases.extend(cases[:n_train])
        eval_cases.extend(cases[n_train:])
        print(f"  {ref_set}: {len(cases)} total -> {n_train} train / {len(cases) - n_train} eval")

    # Optionally subsample external benchmarks in the training set
    benchmark_sets = {"HOMSTRAD", "OXBench", "SABRE"}
    if subsample_benchmarks_to is not None:
        from data.subsample import farthest_point_subsample
        from env.msa_state_features import extract_msa_features

        kept_train = []
        for ref_set in sorted(benchmark_sets):
            ref_cases = [c for c in train_cases if c.ref_set == ref_set]
            if not ref_cases:
                continue
            if len(ref_cases) > subsample_benchmarks_to:
                selected = farthest_point_subsample(
                    ref_cases, extract_msa_features, subsample_benchmarks_to,
                )
                print(
                    f"  {ref_set}: subsampled {len(ref_cases)} -> "
                    f"{len(selected)} train (farthest-point)"
                )
                kept_train.extend(selected)
            else:
                kept_train.extend(ref_cases)

        # Add back all non-benchmark train cases unchanged
        kept_train.extend(c for c in train_cases if c.ref_set not in benchmark_sets)
        train_cases = kept_train

    train_dataset = MSADataset(train_cases)
    eval_dataset = MSADataset(eval_cases)

    print(f"Train: {len(train_dataset)} MSAs, Eval: {len(eval_dataset)} MSAs")
    return train_dataset, eval_dataset
