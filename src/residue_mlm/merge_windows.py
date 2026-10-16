"""Merge overlapping window predictions into one full-chain annotation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from residue_mlm.labels import UNK_ID, LevelVocabulary


def iter_window_offsets(full_len: int, max_len: int, stride: int) -> List[int]:
    """Window start indices (0-based) that cover the full chain."""
    if full_len <= 0:
        return [0]
    if full_len <= max_len:
        return [0]

    stride = max(1, min(stride, max_len))
    offsets: List[int] = []
    offset = 0
    while offset < full_len:
        offsets.append(offset)
        if offset + max_len >= full_len:
            break
        offset += stride

    last_start = full_len - max_len
    if last_start > 0 and offsets[-1] != last_start:
        offsets.append(last_start)
    return offsets


def _majority_vote(values: List[int]) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def merge_window_predictions(
    full_len: int,
    window_preds: List[Tuple[int, Dict[str, List[int]]]],
) -> Dict[str, List[int]]:
    """Merge per-window residue predictions by majority vote at each position.

    Parameters
    ----------
    full_len:
        Total sequence length.
    window_preds:
        List of (left_offset, preds) where preds has keys
        domain_ids, cath_c, cath_a, cath_t, cath_h (lists aligned to window).
    """
    domain_votes: List[List[int]] = [[] for _ in range(full_len)]
    cath_votes = {
        level: [[] for _ in range(full_len)]
        for level in ("cath_c", "cath_a", "cath_t", "cath_h")
    }

    for left_offset, preds in window_preds:
        window_len = len(preds["domain_ids"])
        for local_idx in range(window_len):
            global_idx = left_offset + local_idx
            if global_idx >= full_len:
                continue
            domain_votes[global_idx].append(int(preds["domain_ids"][local_idx]))
            for level in cath_votes:
                cath_votes[level][global_idx].append(int(preds[level][local_idx]))

    is_domain: List[bool] = []
    merged_cath = {level: [] for level in cath_votes}
    for pos in range(full_len):
        votes = domain_votes[pos]
        domain_positive = [v for v in votes if v > 0]
        linker_votes = len(votes) - len(domain_positive)
        is_dom = len(domain_positive) > linker_votes
        is_domain.append(is_dom)

        for level in merged_cath:
            level_votes = cath_votes[level][pos]
            if is_dom and level_votes:
                in_domain_votes = [
                    v
                    for v, d in zip(level_votes, votes)
                    if d > 0 and v not in (0, UNK_ID)
                ]
                pool = in_domain_votes if in_domain_votes else [v for v in level_votes if v not in (0, UNK_ID)]
                merged_cath[level].append(_majority_vote(pool) if pool else UNK_ID)
            else:
                merged_cath[level].append(UNK_ID)

    domain_ids = _assign_contiguous_domain_ids(is_domain, merged_cath)
    return {
        "domain_ids": domain_ids,
        "cath_c": merged_cath["cath_c"],
        "cath_a": merged_cath["cath_a"],
        "cath_t": merged_cath["cath_t"],
        "cath_h": merged_cath["cath_h"],
    }


def _assign_contiguous_domain_ids(
    is_domain: Sequence[bool],
    cath: Dict[str, List[int]],
) -> List[int]:
    """Label contiguous domain runs with ids 1..K (0 = linker)."""
    nres = len(is_domain)
    domain_ids = [0] * nres
    domain_idx = 0
    pos = 0
    while pos < nres:
        if not is_domain[pos]:
            pos += 1
            continue
        domain_idx += 1
        start = pos
        signature = tuple(cath[level][pos] for level in ("cath_c", "cath_a", "cath_t", "cath_h"))
        pos += 1
        while pos < nres and is_domain[pos]:
            next_sig = tuple(cath[level][pos] for level in ("cath_c", "cath_a", "cath_t", "cath_h"))
            if next_sig != signature:
                break
            pos += 1
        for idx in range(start, pos):
            domain_ids[idx] = domain_idx
    return domain_ids

