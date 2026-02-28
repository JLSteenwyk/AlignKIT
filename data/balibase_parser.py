"""Parse BAliBASE reference alignments and extract pairwise references."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


GAP_CHARS = frozenset("-.~")


@dataclass
class PairwiseReference:
    """A pairwise reference alignment extracted from a BAliBASE MSA."""
    seq1_id: str
    seq2_id: str
    seq1_raw: str        # unaligned sequence (no gaps)
    seq2_raw: str        # unaligned sequence (no gaps)
    seq1_aligned: str    # aligned sequence (with gaps)
    seq2_aligned: str    # aligned sequence (with gaps)
    ref_set: str         # e.g. "RV11"
    source_file: str     # path to source alignment file


def _normalize_id(seq_id: str) -> str:
    """Normalize sequence ID for consistent matching."""
    # Remove common suffixes/prefixes added by different parsers
    seq_id = seq_id.strip()
    # Remove trailing /start-end range
    seq_id = re.sub(r"/\d+-\d+$", "", seq_id)
    return seq_id


def _remove_gap_only_columns(seq1: str, seq2: str) -> Tuple[str, str]:
    """Remove columns where both sequences have gaps."""
    new1, new2 = [], []
    for c1, c2 in zip(seq1, seq2):
        if c1 in GAP_CHARS and c2 in GAP_CHARS:
            continue
        new1.append(c1)
        new2.append(c2)
    return "".join(new1), "".join(new2)


def _standardize_gaps(seq: str) -> str:
    """Replace all gap characters with '-'."""
    return re.sub(r"[.~]", "-", seq)


def _extract_raw(aligned_seq: str) -> str:
    """Extract raw (ungapped) sequence."""
    return re.sub(r"[-. ~]", "", aligned_seq)


def parse_xml_alignment(xml_path: Path) -> Optional[MultipleSeqAlignment]:
    """Parse a BAliBASE 3 XML alignment file.

    BAliBASE 3 XML structure:
      <macsim>
        <alignment>
          <sequence>
            <seq-name>...</seq-name>
            <seq-data>...aligned sequence with gaps...</seq-data>
          </sequence>
          ...
        </alignment>
      </macsim>
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    records = []

    # BAliBASE 3 structure: <macsim>/<alignment>/<sequence>
    sequences = root.findall(".//sequence")

    for seq_elem in sequences:
        # Get name from <seq-name> child element
        name = ""
        name_elem = seq_elem.find("seq-name")
        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip()
        else:
            # Fallback: try attribute
            name = seq_elem.get("name", seq_elem.get("id", ""))

        # Get aligned sequence from <seq-data> child element
        aligned_text = ""
        data_elem = seq_elem.find("seq-data")
        if data_elem is not None and data_elem.text:
            aligned_text = data_elem.text.strip()
        else:
            # Fallback: try <aligned> element
            aligned_elem = seq_elem.find("aligned")
            if aligned_elem is not None and aligned_elem.text:
                aligned_text = aligned_elem.text.strip()

        # Clean up: remove whitespace within sequence
        aligned_text = re.sub(r"\s+", "", aligned_text)

        if name and aligned_text:
            record = SeqRecord(Seq(aligned_text), id=_normalize_id(name), description="")
            records.append(record)

    if len(records) >= 2:
        # Ensure equal length
        max_len = max(len(r.seq) for r in records)
        for r in records:
            if len(r.seq) < max_len:
                r.seq = Seq(str(r.seq) + "-" * (max_len - len(r.seq)))
        return MultipleSeqAlignment(records)

    return None


def parse_msf_alignment(msf_path: Path) -> Optional[MultipleSeqAlignment]:
    """Parse an MSF format alignment file using BioPython."""
    try:
        alignment = AlignIO.read(msf_path, "msf")
        return alignment
    except Exception:
        return None


def parse_fasta_alignment(fasta_path: Path) -> Optional[MultipleSeqAlignment]:
    """Parse a FASTA format alignment file."""
    try:
        alignment = AlignIO.read(fasta_path, "fasta")
        return alignment
    except Exception:
        return None


def parse_balibase_alignment(path: Path) -> Optional[MultipleSeqAlignment]:
    """Parse a BAliBASE alignment file, trying XML, MSF, then FASTA."""
    suffix = path.suffix.lower()

    # Try format based on extension
    if suffix == ".xml":
        result = parse_xml_alignment(path)
        if result:
            return result

    if suffix == ".msf":
        result = parse_msf_alignment(path)
        if result:
            return result

    if suffix in (".fasta", ".fa", ".tfa"):
        result = parse_fasta_alignment(path)
        if result:
            return result

    # Try all formats as fallback
    for parser in [parse_xml_alignment, parse_msf_alignment, parse_fasta_alignment]:
        try:
            result = parser(path)
            if result:
                return result
        except Exception:
            continue

    return None


def extract_pairwise_references(
    msa: MultipleSeqAlignment, ref_set: str, source_file: str
) -> List[PairwiseReference]:
    """Extract all N*(N-1)/2 pairwise references from an MSA.

    For each pair, remove double-gap columns and produce aligned + raw sequences.
    """
    n = len(msa)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            seq1_full = _standardize_gaps(str(msa[i].seq))
            seq2_full = _standardize_gaps(str(msa[j].seq))

            # Remove columns where both are gaps
            seq1_aligned, seq2_aligned = _remove_gap_only_columns(seq1_full, seq2_full)

            seq1_raw = _extract_raw(seq1_aligned)
            seq2_raw = _extract_raw(seq2_aligned)

            # Skip if either sequence is empty
            if not seq1_raw or not seq2_raw:
                continue

            pairs.append(PairwiseReference(
                seq1_id=_normalize_id(msa[i].id),
                seq2_id=_normalize_id(msa[j].id),
                seq1_raw=seq1_raw.upper(),
                seq2_raw=seq2_raw.upper(),
                seq1_aligned=seq1_aligned.upper(),
                seq2_aligned=seq2_aligned.upper(),
                ref_set=ref_set,
                source_file=str(source_file),
            ))

    return pairs


def load_all_references(
    data_dir: Path, ref_sets: List[str]
) -> Dict[str, List[PairwiseReference]]:
    """Walk BAliBASE directory tree and load all pairwise references.

    Returns dict mapping ref_set name -> list of PairwiseReference.
    """
    all_refs: Dict[str, List[PairwiseReference]] = {}

    for ref_set in ref_sets:
        ref_dir = data_dir / ref_set
        if not ref_dir.exists():
            print(f"  Warning: {ref_dir} not found, skipping")
            continue

        refs = []
        # BAliBASE 3 files: BB*.msf = full reference, BBS*.msf = core blocks only
        # We use BB* (full reference) MSF files as they are most reliably parsed.
        # .tfa files contain raw (ungapped) input sequences, not alignments.
        msf_files = sorted(ref_dir.glob("BB[0-9]*.msf"))

        if not msf_files:
            # Fallback: try any MSF file
            msf_files = sorted(ref_dir.glob("*.msf"))

        for filepath in msf_files:
            msa = parse_balibase_alignment(filepath)
            if msa is None:
                continue

            pairs = extract_pairwise_references(msa, ref_set, str(filepath))
            refs.extend(pairs)

        all_refs[ref_set] = refs
        print(f"  {ref_set}: loaded {len(refs)} pairwise references from {len(msf_files)} files")

    return all_refs
