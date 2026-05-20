"""Build Stage-A / Stage-B training targets from a ``chopping_star`` string.

The canonical annotation format (mirrors ``benchmark/ted_eval.py``):

    "<seg>_<seg> | <C.A.T.H> * <seg> | <C.A.T.H>"

* domains separated by ``*``
* within a domain, ``boundary | cath_label`` (``-`` = unknown CATH)
* discontinuous segments joined by ``_``
* each segment ``start-end`` is **1-based inclusive** by default

Internally we convert to **0-based half-open** ``[start, end)`` residue intervals
to match the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_SEG_RE = re.compile(r"^(\d+)-(\d+)$")


@dataclass
class ParsedDomain:
    segments: List[Tuple[int, int]]          # 0-based half-open, sorted
    cath: Optional[str] = None               # dotted code or None

    @property
    def residues(self) -> List[int]:
        out: List[int] = []
        for s, e in self.segments:
            out.extend(range(s, e))
        return out


@dataclass
class ParsedChopping:
    domains: List[ParsedDomain] = field(default_factory=list)
    parse_ok: bool = True
    errors: List[str] = field(default_factory=list)


def parse_chopping_star(
    annotation: object,
    *,
    one_based_inclusive: bool = True,
) -> ParsedChopping:
    """Parse a chopping_star string into 0-based half-open domains."""
    parsed = ParsedChopping()
    if annotation is None:
        parsed.parse_ok = False
        return parsed
    text = str(annotation).strip()
    if not text:
        parsed.parse_ok = False
        return parsed

    for chunk_idx, raw in enumerate(text.split("*")):
        chunk = raw.strip()
        if not chunk:
            continue
        if "|" not in chunk:
            parsed.parse_ok = False
            parsed.errors.append(f"missing_pipe_chunk_{chunk_idx}")
            continue
        bounds_text, class_text = chunk.split("|", 1)
        cath = None if class_text.strip() in {"", "-"} else class_text.strip()

        segments: List[Tuple[int, int]] = []
        for seg_raw in bounds_text.split("_"):
            seg = seg_raw.strip()
            if not seg:
                continue
            m = _SEG_RE.match(seg)
            if not m:
                parsed.parse_ok = False
                parsed.errors.append(f"bad_segment_{seg}")
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                parsed.parse_ok = False
                parsed.errors.append(f"descending_{seg}")
                continue
            if one_based_inclusive:
                start, end = a - 1, b          # [a-1, b)  half-open
            else:
                start, end = a, b
            if end <= start:
                parsed.parse_ok = False
                continue
            segments.append((start, end))

        if segments:
            segments.sort()
            parsed.domains.append(ParsedDomain(segments=segments, cath=cath))
        else:
            parsed.parse_ok = False
            parsed.errors.append(f"no_segments_chunk_{chunk_idx}")

    return parsed


def residue_assignment(domains: List[ParsedDomain], nres: int) -> np.ndarray:
    """Per-residue domain id: 0 = linker/NDR, k = (1-based) domain index.

    Residues are assigned to the first domain that claims them (overlaps go to
    the earlier domain), matching ``ted_eval._build_domain_dict``.
    """
    assign = np.zeros(nres, dtype=np.int64)
    for idx, dom in enumerate(domains, start=1):
        for s, e in dom.segments:
            s_c, e_c = max(0, s), min(nres, e)
            for r in range(s_c, e_c):
                if assign[r] == 0:
                    assign[r] = idx
    return assign


def domain_residue_masks(domains: List[ParsedDomain], nres: int) -> List[np.ndarray]:
    """One boolean residue mask per domain (after overlap resolution)."""
    assign = residue_assignment(domains, nres)
    return [assign == idx for idx in range(1, len(domains) + 1)]


def comembership_matrix(assign: np.ndarray) -> np.ndarray:
    """Symmetric (nres, nres) int matrix: 1 iff i,j share a (non-linker) domain.

    Linker residues (id 0) are co-member with nobody (including each other).
    Provided for tests and as the reference Stage-A pairwise target.
    """
    nres = assign.shape[0]
    same = (assign[:, None] == assign[None, :]) & (assign[:, None] > 0)
    return same.astype(np.int64).reshape(nres, nres)


def in_domain_mask(assign: np.ndarray) -> np.ndarray:
    """Boolean per-residue: True if the residue belongs to any domain."""
    return assign > 0
