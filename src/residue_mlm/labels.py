"""Parse chopping_star annotations into per-residue supervision targets."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

DOMAIN_SPLIT_RE = re.compile(r"\s*\*\s*")
SEGMENT_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
CATH_PART_RE = re.compile(r"^\d+$")

PAD_ID = 0
UNK_ID = 1
MASK_ID = 2
SPECIAL_VOCAB_SIZE = 3

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx + SPECIAL_VOCAB_SIZE for idx, aa in enumerate(AA_ALPHABET)}
AA_TO_ID["X"] = UNK_ID
AA_MASK_ID = MASK_ID


def split_domain_tokens(chopping_star: object) -> List[str]:
    if chopping_star is None:
        return []
    text = str(chopping_star).strip()
    if not text:
        return []
    return [token.strip() for token in DOMAIN_SPLIT_RE.split(text) if token.strip()]


def parse_domains_from_annotation(
    chopping_star: object,
    *,
    index_base: int = 1,
) -> List[Dict[str, object]]:
    domains: List[Dict[str, object]] = []
    for token in split_domain_tokens(chopping_star):
        if "|" not in token:
            continue
        left, right = token.split("|", 1)
        cath_label = right.strip()
        if cath_label in {"", "-"}:
            cath_label = None

        segments: List[Tuple[int, int]] = []
        for segment_text in [seg.strip() for seg in left.strip().split("_") if seg.strip()]:
            match = SEGMENT_RE.match(segment_text)
            if not match:
                continue
            start_text, end_text = int(match.group(1)), int(match.group(2))
            if end_text < start_text:
                continue
            start = start_text - index_base
            end = end_text - index_base + 1
            segments.append((start, end))

        if segments:
            domains.append({"segments": segments, "cath_label": cath_label})
    return domains


def split_cath_levels(cath_label: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if cath_label is None:
        return None, None, None, None
    parts = [part for part in cath_label.split(".") if part]
    if not parts or not all(CATH_PART_RE.match(part) for part in parts):
        return None, None, None, None
    padded = (parts + [None, None, None, None])[:4]
    return padded[0], padded[1], padded[2], padded[3]


class LevelVocabulary:
    def __init__(self, name: str):
        self.name = name
        self.token2id: Dict[str, int] = {
            "<pad>": PAD_ID,
            "<unk>": UNK_ID,
            "<mask>": MASK_ID,
        }
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def fit(self, values: List[Optional[str]]) -> "LevelVocabulary":
        for value in values:
            if value is None:
                continue
            if value not in self.token2id:
                idx = len(self.token2id)
                self.token2id[value] = idx
                self.id2token[idx] = value
        return self

    def encode(self, value: Optional[str]) -> int:
        if value is None:
            return UNK_ID
        return self.token2id.get(value, UNK_ID)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def mask_id(self) -> int:
        return MASK_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID


def encode_residue_labels(
    sequence: str,
    chopping_star: object,
    *,
    max_len: int = 128,
    index_base: int = 1,
    left_offset: int = 0,
) -> Dict[str, object]:
    """Build per-residue labels for one cropped chain window."""
    seq = str(sequence).strip().upper()
    full_len = len(seq)
    left_offset = max(0, min(left_offset, max(0, full_len - 1)))
    window_end = min(full_len, left_offset + max_len)
    seq = seq[left_offset:window_end]
    nres = len(seq)

    domains = parse_domains_from_annotation(chopping_star, index_base=index_base)
    num_domains = len(domains)

    domain_ids = [0] * nres
    cath_c: List[Optional[str]] = [None] * nres
    cath_a: List[Optional[str]] = [None] * nres
    cath_t: List[Optional[str]] = [None] * nres
    cath_h: List[Optional[str]] = [None] * nres

    assigned: set[int] = set()
    for domain_idx, domain in enumerate(domains, start=1):
        level_c, level_a, level_t, level_h = split_cath_levels(domain.get("cath_label"))
        residues: set[int] = set()
        for start, end in domain["segments"]:
            start = max(0, start - left_offset)
            end = min(nres, end - left_offset)
            if end <= start:
                continue
            residues.update(range(start, end))

        clean_residues = sorted(r for r in residues if 0 <= r < nres and r not in assigned)
        if not clean_residues:
            continue

        for residue_idx in clean_residues:
            domain_ids[residue_idx] = domain_idx
            cath_c[residue_idx] = level_c
            cath_a[residue_idx] = level_a
            cath_t[residue_idx] = level_t
            cath_h[residue_idx] = level_h
        assigned.update(clean_residues)

    aa_ids = [AA_TO_ID.get(aa, UNK_ID) for aa in seq]
    positions = list(range(nres))

    return {
        "aa_ids": aa_ids,
        "positions": positions,
        "domain_ids": domain_ids,
        "num_domains": num_domains,
        "cath_c": cath_c,
        "cath_a": cath_a,
        "cath_t": cath_t,
        "cath_h": cath_h,
        "seq_len": nres,
        "left_offset": left_offset,
    }
