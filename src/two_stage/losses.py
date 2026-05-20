"""Loss functions for the two-stage model."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def _pos_weight(target: torch.Tensor, valid: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Balanced positive weight = #neg / #pos over the valid entries."""
    pos = (target[valid] > 0.5).sum().clamp_min(1)
    neg = (target[valid] <= 0.5).sum().clamp_min(1)
    return (neg.float() / pos.float()).clamp_max(cap)


def residue_bce_loss(
    residue_logit: torch.Tensor,   # (B, L)
    assign: torch.Tensor,          # (B, L) int, 0=linker, k=domain
    mask: torch.Tensor,            # (B, L) bool, True=real residue
    balance: bool = True,
) -> torch.Tensor:
    target = (assign > 0).float()
    valid = mask
    if valid.sum() == 0:
        return residue_logit.sum() * 0.0
    pw = _pos_weight(target, valid) if balance else None
    loss = F.binary_cross_entropy_with_logits(
        residue_logit[valid], target[valid], pos_weight=pw, reduction="mean"
    )
    return loss


def pairwise_bce_loss(
    pair_logit: torch.Tensor,      # (B, L, L)
    assign: torch.Tensor,          # (B, L) int
    mask: torch.Tensor,            # (B, L) bool
    balance: bool = True,
    max_pair_len: int = 768,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Symmetric co-membership BCE.

    For chains longer than ``max_pair_len`` a random residue subset is used per
    sample to bound the O(L^2) memory/compute (the matrix is symmetric and
    redundant, so a subset is an unbiased estimate).
    """
    B, L, _ = pair_logit.shape
    device = pair_logit.device
    losses = []
    pos_total = neg_total = 0

    # Precompute a shared balanced weight across the batch if requested.
    for b in range(B):
        valid_res = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        n = valid_res.numel()
        if n < 2:
            continue
        if n > max_pair_len:
            # randperm on CPU (matches a CPU generator), then move to the data device.
            perm = torch.randperm(n, generator=generator)[:max_pair_len].to(device)
            idx = valid_res[perm]
        else:
            idx = valid_res
        a = assign[b, idx]                              # (m,)
        same = (a[:, None] == a[None, :]) & (a[:, None] > 0)
        target = same.float()
        logit = pair_logit[b][idx][:, idx]              # (m, m)

        m = idx.numel()
        # exclude the diagonal (i==j) from the loss
        off_diag = ~torch.eye(m, dtype=torch.bool, device=device)
        t = target[off_diag]
        z = logit[off_diag]
        pos_total += int((t > 0.5).sum())
        neg_total += int((t <= 0.5).sum())
        losses.append((z, t))

    if not losses:
        return pair_logit.sum() * 0.0

    pw = None
    if balance:
        pos = max(1, pos_total)
        neg = max(1, neg_total)
        pw = torch.tensor(min(50.0, neg / pos), device=device)

    total = 0.0
    count = 0
    for z, t in losses:
        total = total + F.binary_cross_entropy_with_logits(
            z, t, pos_weight=pw, reduction="sum"
        )
        count += z.numel()
    return total / max(1, count)


def cath_loss(
    cath_logits: List[torch.Tensor],          # 4 x (N, V_level)
    targets: torch.Tensor,                     # (N, 4) int level ids
    *,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    class_weights: Optional[List[Optional[torch.Tensor]]] = None,  # per level (V_level,)
    level_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Hierarchical CATH loss applied to the CATH head ONLY.

    Long-tail handling lives here (class-balanced ``class_weights`` and/or
    ``focal_gamma``) so the boundary objective stays on the natural distribution.

    * ``focal_gamma == 0`` -> (optionally class-weighted, label-smoothed) cross-entropy.
    * ``focal_gamma  > 0`` -> class-weighted focal loss (label smoothing ignored).
    ``level_weights`` lets coarse levels (C, A) be emphasised over fine ones (H),
    e.g. for a coarse-to-fine curriculum.
    """
    if cath_logits is None or targets.numel() == 0:
        return cath_logits[0].sum() * 0.0 if cath_logits else torch.tensor(0.0)

    n_levels = len(cath_logits)
    if level_weights is None:
        level_weights = [1.0] * n_levels

    total = 0.0
    wsum = 0.0
    for level in range(n_levels):
        logits = cath_logits[level]
        tgt = targets[:, level]
        cw = None if class_weights is None else class_weights[level]
        if focal_gamma and focal_gamma > 0:
            logp = F.log_softmax(logits, dim=-1)
            logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            pt = logpt.exp()
            loss = -((1.0 - pt) ** focal_gamma) * logpt
            if cw is not None:
                loss = loss * cw.to(logits.device)[tgt]
            ce = loss.mean()
        else:
            ce = F.cross_entropy(
                logits, tgt,
                weight=None if cw is None else cw.to(logits.device),
                label_smoothing=label_smoothing,
            )
        total = total + level_weights[level] * ce
        wsum += level_weights[level]
    return total / max(1e-8, wsum)


# Backward-compatible alias (plain hierarchical CE).
def cath_ce_loss(cath_logits, targets, label_smoothing: float = 0.0, level_weights=None):
    return cath_loss(
        cath_logits, targets,
        label_smoothing=label_smoothing, level_weights=level_weights,
    )
