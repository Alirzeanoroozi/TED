"""Decode residue-level model outputs into chopping_star annotations."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from residue_mlm.labels import LevelVocabulary, UNK_ID


def ids_to_level_string(token_id: int, vocab: LevelVocabulary) -> Optional[str]:
    if token_id in (0, UNK_ID, vocab.mask_id):
        return None
    token = vocab.id2token.get(int(token_id))
    if token is None or token.startswith("<"):
        return None
    return token


def _majority_string(ids: Sequence[int], vocab: LevelVocabulary) -> Optional[str]:
    decoded = [ids_to_level_string(int(v), vocab) for v in ids]
    decoded = [v for v in decoded if v is not None]
    if not decoded:
        return None
    return Counter(decoded).most_common(1)[0][0]


def _join_cath_parts(
    c: Optional[str],
    a: Optional[str],
    t: Optional[str],
    h: Optional[str],
) -> str:
    parts: List[str] = []
    for part in (c, a, t, h):
        if part is None:
            break
        parts.append(part)
    return ".".join(parts) if parts else "-"


def merge_consecutive_indices(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    segments: List[Tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = idx
        prev = idx
    segments.append((start, prev))
    return segments


def extract_domain_segments(domain_ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Return (start, end_exclusive, domain_id) runs for non-linker residues."""
    segments: List[Tuple[int, int, int]] = []
    nres = len(domain_ids)
    idx = 0
    while idx < nres:
        domain_id = int(domain_ids[idx])
        if domain_id == 0:
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < nres and int(domain_ids[idx]) == domain_id:
            idx += 1
        segments.append((start, idx, domain_id))
    return segments


def residue_predictions_to_chopping_star(
    domain_ids: Sequence[int],
    cath_c: Sequence[int],
    cath_a: Sequence[int],
    cath_t: Sequence[int],
    cath_h: Sequence[int],
    vocabs: Dict[str, LevelVocabulary],
    *,
    left_offset: int = 0,
    index_base: int = 1,
) -> str:
    tokens: List[str] = []
    for start, end, _domain_id in extract_domain_segments(domain_ids):
        segment_texts = []
        for seg_start, seg_end in merge_consecutive_indices(list(range(start, end))):
            start_out = seg_start + left_offset + index_base
            end_out = seg_end + left_offset + index_base
            segment_texts.append(f"{start_out}-{end_out}")

        cath_label = _join_cath_parts(
            _majority_string(cath_c[start:end], vocabs["cath_c"]),
            _majority_string(cath_a[start:end], vocabs["cath_a"]),
            _majority_string(cath_t[start:end], vocabs["cath_t"]),
            _majority_string(cath_h[start:end], vocabs["cath_h"]),
        )
        tokens.append(f"{'_'.join(segment_texts)} | {cath_label}")
    return " * ".join(tokens)


def infer_num_domains(domain_ids: torch.Tensor, seq_len: int) -> int:
    if seq_len <= 0:
        return 0
    active = domain_ids[:seq_len]
    unique = {int(v) for v in active.tolist() if int(v) > 0}
    return len(unique)


def batch_predictions_to_chopping_stars(
    pred_domain_ids: torch.Tensor,
    pred_cath: Dict[str, torch.Tensor],
    seq_lens: torch.Tensor,
    left_offsets: Sequence[int],
    vocabs: Dict[str, LevelVocabulary],
    *,
    index_base: int = 1,
) -> List[str]:
    batch_size = pred_domain_ids.size(0)
    outputs: List[str] = []
    for batch_idx in range(batch_size):
        length = int(seq_lens[batch_idx])
        outputs.append(
            residue_predictions_to_chopping_star(
                pred_domain_ids[batch_idx, :length].tolist(),
                pred_cath["cath_c"][batch_idx, :length].tolist(),
                pred_cath["cath_a"][batch_idx, :length].tolist(),
                pred_cath["cath_t"][batch_idx, :length].tolist(),
                pred_cath["cath_h"][batch_idx, :length].tolist(),
                vocabs,
                left_offset=int(left_offsets[batch_idx]),
                index_base=index_base,
            )
        )
    return outputs
