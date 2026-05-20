"""Serialize predicted domains back to a canonical ``chopping_star`` string.

This is the bridge that lets the two-stage model feed the existing evaluator
(``benchmark/ted_eval.py``) and benchmark runner without changes: predictions
become the same ``start-end | C.A.T.H * ...`` strings the old model emitted.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def assignment_to_segments(assign: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Per-domain list of contiguous 0-based half-open segments.

    Domains are ordered by their (already-assigned) 1..K id; a domain split into
    several runs of residues yields several segments (discontinuous domain).
    """
    n_dom = int(assign.max()) if assign.size else 0
    segments_per_domain: List[List[Tuple[int, int]]] = [[] for _ in range(n_dom)]
    if n_dom == 0:
        return segments_per_domain

    L = assign.shape[0]
    r = 0
    while r < L:
        lab = int(assign[r])
        if lab == 0:
            r += 1
            continue
        start = r
        while r < L and int(assign[r]) == lab:
            r += 1
        segments_per_domain[lab - 1].append((start, r))  # [start, r) half-open
    return segments_per_domain


def domains_to_chopping_star(
    segments_per_domain: Sequence[Sequence[Tuple[int, int]]],
    cath_per_domain: Optional[Sequence[Optional[str]]] = None,
    *,
    one_based_inclusive: bool = True,
) -> str:
    """Build a ``chopping_star`` string from per-domain segments + CATH codes."""
    if cath_per_domain is None:
        cath_per_domain = [None] * len(segments_per_domain)

    domain_tokens: List[str] = []
    for segments, cath in zip(segments_per_domain, cath_per_domain):
        if not segments:
            continue
        seg_strs: List[str] = []
        for s, e in segments:
            if one_based_inclusive:
                a, b = s + 1, e            # half-open [s,e) -> inclusive s+1..e
            else:
                a, b = s, e
            seg_strs.append(f"{a}-{b}")
        label = cath if cath not in (None, "", "-") else "-"
        domain_tokens.append(f"{'_'.join(seg_strs)} | {label}")

    return " * ".join(domain_tokens)


def assignment_and_cath_to_chopping_star(
    assign: np.ndarray,
    cath_per_domain: Optional[Sequence[Optional[str]]] = None,
    *,
    one_based_inclusive: bool = True,
) -> str:
    """Convenience: assignment array + per-domain CATH -> chopping_star string."""
    segs = assignment_to_segments(assign)
    if cath_per_domain is not None and len(cath_per_domain) != len(segs):
        # Be lenient: pad/truncate so a length mismatch never crashes inference.
        cath_per_domain = list(cath_per_domain)[: len(segs)] + [None] * max(
            0, len(segs) - len(cath_per_domain)
        )
    return domains_to_chopping_star(
        segs, cath_per_domain, one_based_inclusive=one_based_inclusive
    )
