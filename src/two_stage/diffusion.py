"""Small SCISOR-style insertion/deletion utilities for TED.

The TED adaptation uses domain residues as the clean sequence and treats
inserted/non-domain residues as deletion targets.  These helpers intentionally
stay lightweight: they generate artificial inserted residues with the same
negative-binomial length schedule used by SCISOR, and they return the exact
sampled insertion mask as the deletion-posterior target for v1 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class NoisedSequence:
    sequence: str
    inserted_mask: List[bool]
    clean_length: int
    insertions: int
    t: float


def domain_only_sequence(sequence: str, assign: torch.Tensor, mask: torch.Tensor) -> str:
    """Return residues covered by a TED domain, preserving sequence order."""
    chars = []
    n = min(len(sequence), int(mask.sum().item()), assign.shape[0])
    assign_cpu = assign[:n].detach().cpu()
    mask_cpu = mask[:n].detach().cpu()
    for i in range(n):
        if bool(mask_cpu[i]) and int(assign_cpu[i]) > 0:
            chars.append(sequence[i])
    return "".join(chars)


def _sample_insertions(
    clean_len: int,
    *,
    t_max: float,
    alpha_rate: float,
    max_insertions: int,
    max_noised_len: int,
    generator: torch.Generator | None = None,
) -> tuple[int, float]:
    if clean_len <= 0:
        return 0, 0.0
    max_allowed = min(max_insertions, max(0, max_noised_len - clean_len))
    if max_allowed <= 0:
        return 0, 0.0

    t = float(torch.rand((), generator=generator).item() * t_max)
    alpha = (1.0 - t) ** (1.0 / alpha_rate)
    dist = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(float(clean_len + 1)),
        probs=torch.tensor(float(1.0 - alpha)),
    )
    insertions = int(dist.sample().item())
    # Rao-Blackwellized SCISOR training ignores zero-insertion examples.  For this
    # auxiliary batch, force at least one inserted residue so every sample has a
    # deletion target.
    insertions = max(1, min(insertions, max_allowed))
    return insertions, t


def _random_aas(n: int, generator: torch.Generator | None = None) -> list[str]:
    idx = torch.randint(0, len(AA_ALPHABET), (n,), generator=generator).tolist()
    return [AA_ALPHABET[i] for i in idx]


def insert_random_residues(
    clean: str,
    insertions: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[str, list[bool]]:
    """Insert random AAs uniformly among ``len(clean)+1`` insertion sites."""
    L = len(clean)
    if insertions <= 0:
        return clean, [False] * L

    bins = torch.randint(0, L + 1, (insertions,), generator=generator).tolist()
    residues = _random_aas(insertions, generator=generator)
    by_bin: list[list[str]] = [[] for _ in range(L + 1)]
    for site, aa in zip(bins, residues):
        by_bin[site].append(aa)

    out_chars: list[str] = []
    inserted_mask: list[bool] = []
    for site in range(L + 1):
        for aa in by_bin[site]:
            out_chars.append(aa)
            inserted_mask.append(True)
        if site < L:
            out_chars.append(clean[site])
            inserted_mask.append(False)
    return "".join(out_chars), inserted_mask


def make_artificial_insertion_batch(
    sequences: Sequence[str],
    assign: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_noised_len: int,
    t_max: float = 0.9,
    alpha_rate: float = 1.1,
    max_insertions: int = 128,
    generator: torch.Generator | None = None,
) -> list[NoisedSequence]:
    """Create artificial noised domain-core sequences for the deletion loss."""
    out: list[NoisedSequence] = []
    for b, sequence in enumerate(sequences):
        clean = domain_only_sequence(sequence, assign[b], mask[b])
        if not clean:
            continue
        insertions, t = _sample_insertions(
            len(clean),
            t_max=t_max,
            alpha_rate=alpha_rate,
            max_insertions=max_insertions,
            max_noised_len=max_noised_len,
            generator=generator,
        )
        if insertions <= 0:
            continue
        noised, inserted_mask = insert_random_residues(clean, insertions, generator=generator)
        out.append(
            NoisedSequence(
                sequence=noised,
                inserted_mask=inserted_mask,
                clean_length=len(clean),
                insertions=insertions,
                t=t,
            )
        )
    return out


def pad_inserted_masks(
    examples: Sequence[NoisedSequence],
    *,
    device: torch.device,
) -> torch.Tensor:
    Lmax = max((len(ex.inserted_mask) for ex in examples), default=0)
    target = torch.zeros(len(examples), Lmax, dtype=torch.bool, device=device)
    for i, ex in enumerate(examples):
        target[i, : len(ex.inserted_mask)] = torch.tensor(
            ex.inserted_mask, dtype=torch.bool, device=device
        )
    return target


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid residue positions only."""
    return F.log_softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)


def centered_deletion_evidence(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Length-normalized deletion evidence for fusion/metrics.

    Values above zero are more deletion-like than the chain average; values below
    zero are more domain-like than average.
    """
    logp = masked_log_softmax(logits, mask)
    valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (logp.masked_fill(~mask, 0.0).sum(dim=1, keepdim=True) / valid_counts)
    return (logp - mean).masked_fill(~mask, 0.0)
