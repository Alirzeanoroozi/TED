"""Decode domain-query outputs into a chopping_star string + one confidence per domain."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from residue_mlm.decode import _join_cath_parts, ids_to_level_string, merge_consecutive_indices
from residue_mlm.labels import LevelVocabulary

CATH_LEVELS = ("cath_c", "cath_a", "cath_t", "cath_h")


def get_runs(domain_ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Return (start, end_exclusive, domain_id) for every run of equal ids, including 0 (no domain)."""
    runs = []
    start = 0
    for idx in range(1, len(domain_ids) + 1):
        if idx == len(domain_ids) or domain_ids[idx] != domain_ids[start]:
            runs.append((start, idx, domain_ids[start]))
            start = idx
    return runs


def fill_small_gaps(domain_ids: List[int], max_gap: int) -> List[int]:
    """A short run between two pieces of the same domain becomes that domain: 1 1 0 1 1 -> 1 1 1 1 1."""
    runs = get_runs(domain_ids)
    for (_, _, left), (start, end, _), (_, _, right) in zip(runs, runs[1:], runs[2:]):
        if left == right != 0 and end - start <= max_gap:
            domain_ids[start:end] = [left] * (end - start)
    return domain_ids


def remove_short_segments(domain_ids: List[int], min_segment_len: int) -> List[int]:
    """Domain pieces shorter than min_segment_len become 0 (no domain)."""
    for start, end, domain_id in get_runs(domain_ids):
        if domain_id != 0 and end - start < min_segment_len:
            domain_ids[start:end] = [0] * (end - start)
    return domain_ids


def decode_domains(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    cath_logits: Dict[str, torch.Tensor],
    seq_len: int,
    vocabs: Dict[str, LevelVocabulary],
    *,
    score_threshold: float = 0.5,
    max_gap: int = 15,
    min_segment_len: int = 20,
    left_offset: int = 0,
    index_base: int = 1,
) -> Tuple[str, List[float]]:
    """Outputs for ONE protein (Q x L, Q, Q x vocab) -> ("11-50_60-120 | 3.40.50.300 * ...", [0.93, ...])."""
    scores = score_logits.float().sigmoid()
    probs = mask_logits[:, :seq_len].float().sigmoid() * (scores >= score_threshold).unsqueeze(1)

    # Each residue goes to the confident query with the highest mask probability (query k -> id k + 1)
    best_prob, best_query = probs.max(dim=0)
    domain_ids = torch.where(best_prob > 0.5, best_query + 1, torch.zeros_like(best_query)).tolist()

    domain_ids = fill_small_gaps(domain_ids, max_gap)
    domain_ids = remove_short_segments(domain_ids, min_segment_len)

    tokens: List[str] = []
    confidences: List[float] = []
    for domain_id in sorted(set(domain_ids) - {0}, key=domain_ids.index):  # order by first residue
        query = domain_id - 1
        residues = [idx for idx, value in enumerate(domain_ids) if value == domain_id]
        segment_text = "_".join(
            f"{start + left_offset + index_base}-{end + left_offset + index_base}"
            for start, end in merge_consecutive_indices(residues)
        )
        cath_label = _join_cath_parts(
            *(ids_to_level_string(int(cath_logits[level][query].argmax()), vocabs[level]) for level in CATH_LEVELS)
        )
        tokens.append(f"{segment_text} | {cath_label}")
        confidences.append(round(float(scores[query]), 3))
    return " * ".join(tokens), confidences


def decode_batch(
    outputs: Dict[str, torch.Tensor],
    seq_lens: torch.Tensor,
    left_offsets: Sequence[int],
    vocabs: Dict[str, LevelVocabulary],
    **decode_kwargs,
) -> Tuple[List[str], List[List[float]]]:
    chopping_stars, confidences = [], []
    for batch_idx, seq_len in enumerate(seq_lens.tolist()):
        chopping_star, domain_confidences = decode_domains(
            outputs["mask_logits"][batch_idx],
            outputs["score_logits"][batch_idx],
            {level: outputs[f"{level}_logits"][batch_idx] for level in CATH_LEVELS},
            seq_len,
            vocabs,
            left_offset=int(left_offsets[batch_idx]),
            **decode_kwargs,
        )
        chopping_stars.append(chopping_star)
        confidences.append(domain_confidences)
    return chopping_stars, confidences
