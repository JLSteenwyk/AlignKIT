"""Loaders for external MSA benchmark datasets: OXBench, SABRE, HOMSTRAD.

Each loader produces MSATestCase objects compatible with the existing pipeline.
"""

from pathlib import Path
from typing import List, Optional

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data.balibase_parser import _normalize_id
from data.msa_dataset import MSATestCase


def _parse_aligned_fasta(path: Path, gap_chars: str = "-.~") -> Optional[MultipleSeqAlignment]:
    """Parse an aligned FASTA file into a MultipleSeqAlignment.

    Handles various gap characters and normalizes to '-'.

    Args:
        path: Path to the aligned FASTA file.
        gap_chars: Characters to treat as gaps.

    Returns:
        MultipleSeqAlignment or None if parsing fails.
    """
    records = []
    current_id = None
    current_seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    seq_str = "".join(current_seq_parts).upper()
                    # Normalize gap characters to '-'
                    for c in gap_chars:
                        if c != "-":
                            seq_str = seq_str.replace(c, "-")
                    records.append(SeqRecord(Seq(seq_str), id=current_id, description=""))
                current_id = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Last record
    if current_id is not None:
        seq_str = "".join(current_seq_parts).upper()
        for c in gap_chars:
            if c != "-":
                seq_str = seq_str.replace(c, "-")
        records.append(SeqRecord(Seq(seq_str), id=current_id, description=""))

    if len(records) < 2:
        return None

    # Verify all sequences have the same length
    lengths = set(len(r.seq) for r in records)
    if len(lengths) != 1:
        return None

    return MultipleSeqAlignment(records)


def _parse_unaligned_fasta(path: Path) -> tuple:
    """Parse an unaligned FASTA file.

    Returns:
        (seq_ids, sequences) tuple, where seq_ids are raw IDs and
        sequences are ungapped strings.
    """
    seq_ids = []
    sequences = []
    current_id = None
    current_seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    raw_seq = "".join(current_seq_parts).upper()
                    raw_seq = raw_seq.replace("-", "").replace(".", "").replace("~", "")
                    seq_ids.append(current_id)
                    sequences.append(raw_seq)
                current_id = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    if current_id is not None:
        raw_seq = "".join(current_seq_parts).upper()
        raw_seq = raw_seq.replace("-", "").replace(".", "").replace("~", "")
        seq_ids.append(current_id)
        sequences.append(raw_seq)

    return seq_ids, sequences


def load_oxbench_cases(bench_dir: Path) -> List[MSATestCase]:
    """Load OXBench test cases from Edgar's bench bundle.

    Format: bench1.0/ox/in/<id> (unaligned FASTA)
            bench1.0/ox/ref/<id> (reference: '.'=gap, upper=core, lower=non-core)

    Args:
        bench_dir: Path to bench1.0/ox directory.

    Returns:
        List of MSATestCase objects.
    """
    return _load_edgar_bench(bench_dir, dataset_name="OXBench")


def load_sabre_cases(bench_dir: Path) -> List[MSATestCase]:
    """Load SABRE test cases from Edgar's bench bundle.

    Same format as OXBench.

    Args:
        bench_dir: Path to bench1.0/sabre directory.

    Returns:
        List of MSATestCase objects.
    """
    return _load_edgar_bench(bench_dir, dataset_name="SABRE")


def _load_edgar_bench(bench_dir: Path, dataset_name: str) -> List[MSATestCase]:
    """Load test cases from an Edgar-format benchmark directory.

    Expects subdirectories: in/ (unaligned), ref/ (reference alignments).
    Reference format: '.' for gaps, mixed case (upper=core, lower=non-core).
    We convert everything to uppercase with '-' gaps.
    """
    in_dir = bench_dir / "in"
    ref_dir = bench_dir / "ref"

    if not in_dir.exists() or not ref_dir.exists():
        print(f"  Warning: {bench_dir} missing in/ or ref/ subdirectory, skipping")
        return []

    test_cases = []
    input_files = sorted(in_dir.iterdir())

    for input_path in input_files:
        if input_path.is_dir():
            continue
        case_id = input_path.name
        ref_path = ref_dir / case_id
        if not ref_path.exists():
            continue

        # Parse unaligned input
        seq_ids, sequences = _parse_unaligned_fasta(input_path)
        if len(seq_ids) < 2:
            continue

        # Parse reference alignment
        ref_alignment = _parse_aligned_fasta(ref_path, gap_chars="-.~")
        if ref_alignment is None:
            continue

        # Verify ID overlap between input and reference
        input_norm_ids = {_normalize_id(sid) for sid in seq_ids}
        ref_norm_ids = {_normalize_id(r.id) for r in ref_alignment}
        common = input_norm_ids & ref_norm_ids
        if len(common) < 2:
            continue

        test_cases.append(MSATestCase(
            case_id=f"{dataset_name}_{case_id}",
            ref_set=dataset_name,
            fasta_path=input_path,
            msf_path=ref_path,  # reusing field for ref alignment path
            num_sequences=len(seq_ids),
            sequences=sequences,
            seq_ids=[_normalize_id(sid) for sid in seq_ids],
            ref_alignment=ref_alignment,
        ))

    print(f"  {dataset_name}: loaded {len(test_cases)} MSA test cases from {len(input_files)} files")
    return test_cases


def load_homstrad_cases(homstrad_dir: Path) -> List[MSATestCase]:
    """Load HOMSTRAD test cases.

    Format: homstrad/in/<family>.faa  (unaligned FASTA, anonymized IDs)
            homstrad/ref/<family>.faa (reference alignment with '-' gaps)

    Args:
        homstrad_dir: Path to homstrad/ directory with in/ and ref/ subdirs.

    Returns:
        List of MSATestCase objects.
    """
    in_dir = homstrad_dir / "in"
    ref_dir = homstrad_dir / "ref"

    if not in_dir.exists() or not ref_dir.exists():
        print(f"  Warning: {homstrad_dir} missing in/ or ref/ subdirectory, skipping")
        return []

    test_cases = []
    ref_files = sorted(ref_dir.glob("*.faa"))

    for ref_path in ref_files:
        family_name = ref_path.stem
        input_path = in_dir / ref_path.name

        if not input_path.exists():
            continue

        # Parse unaligned input
        seq_ids, sequences = _parse_unaligned_fasta(input_path)
        if len(seq_ids) < 2:
            continue

        # Parse reference alignment
        ref_alignment = _parse_aligned_fasta(ref_path, gap_chars="-")
        if ref_alignment is None:
            continue

        # Verify ID overlap
        input_norm_ids = {_normalize_id(sid) for sid in seq_ids}
        ref_norm_ids = {_normalize_id(r.id) for r in ref_alignment}
        common = input_norm_ids & ref_norm_ids
        if len(common) < 2:
            continue

        test_cases.append(MSATestCase(
            case_id=f"HOMSTRAD_{family_name}",
            ref_set="HOMSTRAD",
            fasta_path=input_path,
            msf_path=ref_path,
            num_sequences=len(seq_ids),
            sequences=sequences,
            seq_ids=[_normalize_id(sid) for sid in seq_ids],
            ref_alignment=ref_alignment,
        ))

    print(f"  HOMSTRAD: loaded {len(test_cases)} MSA test cases from {len(ref_files)} families")
    return test_cases


def load_all_benchmarks(benchmark_dir: Path) -> List[MSATestCase]:
    """Load all available benchmark datasets.

    Expects the following directory structure:
        benchmark_dir/
        ├── bench1.0/ox/     (OXBench)
        ├── bench1.0/sabre/  (SABRE)
        └── homstrad/        (HOMSTRAD with in/ and ref/ subdirs)

    Args:
        benchmark_dir: Path to the benchmarks directory.

    Returns:
        List of all MSATestCase objects from available benchmarks.
    """
    all_cases = []

    ox_dir = benchmark_dir / "bench1.0" / "ox"
    if ox_dir.exists():
        all_cases.extend(load_oxbench_cases(ox_dir))

    sabre_dir = benchmark_dir / "bench1.0" / "sabre"
    if sabre_dir.exists():
        all_cases.extend(load_sabre_cases(sabre_dir))

    homstrad_dir = benchmark_dir / "homstrad"
    if homstrad_dir.exists():
        all_cases.extend(load_homstrad_cases(homstrad_dir))

    return all_cases
