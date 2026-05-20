"""Decode Stage-A predictions into discrete domains.

Inputs (numpy):
    pair_prob       : (L, L) symmetric P(residue i, j in the same domain)
    in_domain_prob  : (L,)   P(residue belongs to any domain)

Output:
    assign : (L,) int64, 0 = linker/NDR, k = (1-based) domain id ordered by first
             residue.

Default strategy is **threshold + connected components** (dependency-free): keep
residues whose in-domain prob clears ``domain_threshold``, add an edge between
two kept residues when their pair prob clears ``pair_threshold``, take connected
components as domains, then drop components shorter than ``min_domain_len``.

``method="spectral"`` uses scikit-learn spectral clustering with an eigengap
heuristic when sklearn is importable; it handles chained/transitive merges
better than connected components but is optional.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _relabel_by_first_residue(raw_assign: np.ndarray) -> np.ndarray:
    """Renumber non-zero labels to 1..K in order of first appearance."""
    out = np.zeros_like(raw_assign)
    next_id = 0
    remap: dict[int, int] = {}
    for r, lab in enumerate(raw_assign):
        if lab == 0:
            continue
        if lab not in remap:
            next_id += 1
            remap[lab] = next_id
        out[r] = remap[lab]
    return out


def connected_components_domains(
    pair_prob: np.ndarray,
    in_domain_prob: np.ndarray,
    *,
    pair_threshold: float = 0.5,
    domain_threshold: float = 0.5,
    min_domain_len: int = 20,
) -> np.ndarray:
    L = in_domain_prob.shape[0]
    kept = np.where(in_domain_prob >= domain_threshold)[0]
    assign = np.zeros(L, dtype=np.int64)
    if kept.size == 0:
        return assign

    uf = _UnionFind(L)
    # Only union kept residues that are confidently co-member.
    sub = pair_prob[np.ix_(kept, kept)] >= pair_threshold
    ki = kept.tolist()
    for a in range(len(ki)):
        # vectorised neighbour scan for residue ki[a]
        neigh = kept[sub[a]]
        for b in neigh:
            if b > ki[a]:
                uf.union(ki[a], int(b))

    raw = np.zeros(L, dtype=np.int64)
    for r in kept:
        raw[r] = uf.find(int(r)) + 1  # +1 so root 0 isn't confused with linker

    # Drop components shorter than min_domain_len -> linker.
    labels, counts = np.unique(raw[raw > 0], return_counts=True)
    too_small = set(int(l) for l, c in zip(labels, counts) if c < min_domain_len)
    for r in range(L):
        if raw[r] in too_small:
            raw[r] = 0

    assign = _relabel_by_first_residue(raw)
    return assign


def _estimate_k(affinity: np.ndarray, max_k: int) -> int:
    """Eigengap heuristic on the normalised Laplacian."""
    deg = affinity.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    lap = np.eye(affinity.shape[0]) - (affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :])
    eig = np.sort(np.linalg.eigvalsh(lap))
    if eig.size <= 1:
        return 1
    gaps = np.diff(eig[: max_k + 1])
    return int(np.argmax(gaps) + 1)


def spectral_domains(
    pair_prob: np.ndarray,
    in_domain_prob: np.ndarray,
    *,
    domain_threshold: float = 0.5,
    min_domain_len: int = 20,
    max_k: int = 12,
) -> np.ndarray:
    try:
        from sklearn.cluster import SpectralClustering
    except Exception:
        return connected_components_domains(
            pair_prob, in_domain_prob,
            domain_threshold=domain_threshold, min_domain_len=min_domain_len,
        )

    L = in_domain_prob.shape[0]
    kept = np.where(in_domain_prob >= domain_threshold)[0]
    assign = np.zeros(L, dtype=np.int64)
    if kept.size < 2:
        if kept.size == 1:
            assign[kept[0]] = 1
        return assign

    aff = pair_prob[np.ix_(kept, kept)].astype(float)
    aff = np.clip((aff + aff.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(aff, 1.0)

    k = max(1, min(_estimate_k(aff, max_k=min(max_k, kept.size - 1)), kept.size))
    if k == 1:
        assign[kept] = 1
    else:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="discretize", random_state=0)
        lab = sc.fit_predict(aff)
        raw = np.zeros(L, dtype=np.int64)
        raw[kept] = lab + 1
        labels, counts = np.unique(raw[raw > 0], return_counts=True)
        too_small = set(int(l) for l, c in zip(labels, counts) if c < min_domain_len)
        for r in range(L):
            if raw[r] in too_small:
                raw[r] = 0
        assign = _relabel_by_first_residue(raw)
        return assign

    return _relabel_by_first_residue(assign)


def comembership_to_domains(
    pair_prob: np.ndarray,
    in_domain_prob: np.ndarray,
    *,
    method: str = "connected_components",
    pair_threshold: float = 0.5,
    domain_threshold: float = 0.5,
    min_domain_len: int = 20,
    max_k: int = 12,
) -> np.ndarray:
    if method == "spectral":
        return spectral_domains(
            pair_prob, in_domain_prob,
            domain_threshold=domain_threshold,
            min_domain_len=min_domain_len, max_k=max_k,
        )
    return connected_components_domains(
        pair_prob, in_domain_prob,
        pair_threshold=pair_threshold,
        domain_threshold=domain_threshold,
        min_domain_len=min_domain_len,
    )
