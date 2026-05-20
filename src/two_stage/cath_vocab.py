"""Hierarchical CATH label vocabulary.

A CATH superfamily code such as ``3.40.50.300`` has four nested levels:

    C = class            -> "3"
    A = architecture     -> "3.40"
    T = topology         -> "3.40.50"
    H = homologous sfam  -> "3.40.50.300"

Instead of generating these digits as free text (the old failure mode), Stage B
classifies each domain with four softmax heads, one per level.  Each level's
token is the *cumulative* dotted prefix, so the level-2 vocabulary entry for
``3.40.50.300`` is ``"3.40"``.  This lets the model exploit the hierarchy and
gives partial credit (``cath_level_score``) naturally.

Reserved id 0 at every level is ``UNK`` and represents a missing/unknown label
(the ``-`` placeholder used in the chopping_star format) or any code not seen in
training.  Predicting UNK is how the model abstains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

UNK_TOKEN = "<unk>"
UNK_ID = 0
N_LEVELS = 4


def split_cath_code(code: Optional[str]) -> List[str]:
    """Return the cumulative dotted prefixes of a CATH code.

    ``"3.40.50.300"`` -> ``["3", "3.40", "3.40.50", "3.40.50.300"]``.
    A missing/unknown/short code yields fewer (or zero) prefixes; callers map
    the absent levels to UNK.

    Returns at most :data:`N_LEVELS` prefixes.
    """
    if code is None:
        return []
    text = str(code).strip()
    if not text or text == "-":
        return []
    parts = [p for p in text.split(".") if p != ""]
    prefixes: List[str] = []
    for i in range(min(len(parts), N_LEVELS)):
        prefixes.append(".".join(parts[: i + 1]))
    return prefixes


class CathVocab:
    """Four cumulative-prefix vocabularies (one per CATH level)."""

    def __init__(
        self,
        level_token2id: Optional[List[Dict[str, int]]] = None,
        level_counts: Optional[List[List[int]]] = None,
    ):
        if level_token2id is None:
            level_token2id = [{UNK_TOKEN: UNK_ID} for _ in range(N_LEVELS)]
        assert len(level_token2id) == N_LEVELS
        self.level_token2id: List[Dict[str, int]] = level_token2id
        self.level_id2token: List[Dict[int, str]] = [
            {i: t for t, i in m.items()} for m in level_token2id
        ]
        # Per-level training-frequency of each class id (index = id). Used to
        # build class-balanced / focal CATH-loss weights. Filled by build() or
        # count_levels(); zeros until then.
        if level_counts is None:
            level_counts = [[0] * len(m) for m in level_token2id]
        self.level_counts: List[List[int]] = level_counts

    # ---- construction --------------------------------------------------- #
    @classmethod
    def build(cls, cath_codes: Sequence[Optional[str]], min_count: int = 1) -> "CathVocab":
        """Build vocabularies from an iterable of CATH code strings.

        ``min_count`` drops level-tokens seen fewer than ``min_count`` times
        (they fall back to UNK), which keeps the long tail from exploding the
        head size while still letting frequent superfamilies be predicted.
        """
        counts: List[Dict[str, int]] = [dict() for _ in range(N_LEVELS)]
        for code in cath_codes:
            for level, prefix in enumerate(split_cath_code(code)):
                counts[level][prefix] = counts[level].get(prefix, 0) + 1

        level_token2id: List[Dict[str, int]] = []
        for level in range(N_LEVELS):
            token2id = {UNK_TOKEN: UNK_ID}
            # deterministic ordering: by descending frequency then token
            ordered = sorted(counts[level].items(), key=lambda kv: (-kv[1], kv[0]))
            for token, c in ordered:
                if c >= min_count:
                    token2id[token] = len(token2id)
            level_token2id.append(token2id)
        vocab = cls(level_token2id)
        # Count per-class training frequency (encoding every code, so UNK -- from
        # dropped/rare codes and from None/'-' unlabeled domains -- is counted too).
        vocab.level_counts = vocab.count_levels(cath_codes)
        return vocab

    def count_levels(self, cath_codes: Sequence[Optional[str]]) -> List[List[int]]:
        """Per-level, per-class-id training counts for the given codes."""
        counts = [[0] * len(m) for m in self.level_token2id]
        for code in cath_codes:
            for level, cid in enumerate(self.encode(code)):
                counts[level][cid] += 1
        return counts

    def class_balanced_weights(self, beta: float = 0.999) -> List[List[float]]:
        """Class-balanced weights (Cui et al. 2019): w_c = (1-beta)/(1-beta^n_c).

        Normalised per level so the mean weight is 1. Classes with zero training
        count get the level's mean weight (neutral).
        """
        out: List[List[float]] = []
        for level in range(N_LEVELS):
            counts = np.asarray(self.level_counts[level], dtype=np.float64)
            eff = 1.0 - np.power(beta, np.maximum(counts, 1.0))
            w = (1.0 - beta) / np.maximum(eff, 1e-12)
            w[counts <= 0] = np.nan  # neutralise unseen classes below
            mean_w = np.nanmean(w) if np.isfinite(np.nanmean(w)) else 1.0
            w = np.where(np.isnan(w), mean_w, w)
            w = w * (len(w) / max(w.sum(), 1e-12))  # normalise: mean weight = 1
            out.append(w.tolist())
        return out

    # ---- encode / decode ------------------------------------------------ #
    def encode(self, code: Optional[str]) -> Tuple[int, int, int, int]:
        """Map a CATH code to a 4-tuple of per-level ids.

        Once a level falls back to UNK (token unseen), every deeper level is UNK
        too, because a child code is meaningless without its parent.
        """
        ids = [UNK_ID, UNK_ID, UNK_ID, UNK_ID]
        prefixes = split_cath_code(code)
        for level, prefix in enumerate(prefixes):
            tid = self.level_token2id[level].get(prefix, UNK_ID)
            if tid == UNK_ID:
                break
            ids[level] = tid
        return tuple(ids)  # type: ignore[return-value]

    def decode(self, ids: Sequence[int]) -> str:
        """Map a 4-tuple of ids back to a dotted CATH code (or ``"-"``).

        Decoding stops at the first UNK level; the deepest known prefix is the
        returned code.  All-UNK decodes to ``"-"`` (no label).
        """
        deepest = "-"
        for level in range(min(N_LEVELS, len(ids))):
            tid = int(ids[level])
            if tid == UNK_ID:
                break
            token = self.level_id2token[level].get(tid)
            if token is None:
                break
            deepest = token
        return deepest

    # ---- sizes / io ----------------------------------------------------- #
    @property
    def level_sizes(self) -> List[int]:
        return [len(m) for m in self.level_token2id]

    def to_dict(self) -> dict:
        return {
            "level_token2id": self.level_token2id,
            "level_counts": self.level_counts,
            "n_levels": N_LEVELS,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CathVocab":
        return cls(
            level_token2id=[{k: int(v) for k, v in m.items()} for m in d["level_token2id"]],
            level_counts=d.get("level_counts"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CathVocab":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return f"CathVocab(level_sizes={self.level_sizes})"
