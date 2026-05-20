"""Frozen ESM2 protein-language-model backbone.

Wraps a pretrained ESM2 model (``fair-esm``) and returns per-residue embeddings
for a batch of raw amino-acid strings.  The encoder is frozen and always reloaded
from the pretrained checkpoint, so:

* it is **not** registered as a submodule (held in a list) -> its ~650M weights
  never enter the LightningModule ``state_dict`` / optimizer, keeping checkpoints
  small and ensuring "sequence-only at inference" with no extra trainable encoder;
* sequences that fit the context window are embedded in a single batched forward;
* longer sequences are embedded in overlapping windows and averaged in the
  overlap, so long multi-domain chains are not silently truncated.

Embeddings are returned aligned to residues ``0..L-1`` (BOS/EOS stripped).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

# Embedding dim and default repr layer for the common ESM2 sizes.
_ESM2_DIMS = {
    "esm2_t48_15B_UR50D": (5120, 48),
    "esm2_t36_3B_UR50D": (2560, 36),
    "esm2_t33_650M_UR50D": (1280, 33),
    "esm2_t30_150M_UR50D": (640, 30),
    "esm2_t12_35M_UR50D": (480, 12),
    "esm2_t6_8M_UR50D": (320, 6),
}


class ESM2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int | None = None,
        chunk_size: int = 1022,
        chunk_overlap: int = 256,
        dtype: str = "bf16",  # "bf16" | "fp16" | "fp32"
    ):
        super().__init__()
        try:
            import esm  # fair-esm
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ESM2Backbone requires the 'fair-esm' package. Install with "
                "`pip install fair-esm` in the training environment."
            ) from exc

        if model_name not in _ESM2_DIMS:
            raise ValueError(
                f"Unknown ESM2 model '{model_name}'. Known: {sorted(_ESM2_DIMS)}"
            )
        self.model_name = model_name
        self._embed_dim, n_layers = _ESM2_DIMS[model_name]
        self.repr_layer = repr_layer if repr_layer is not None else n_layers

        model, alphabet = getattr(esm.pretrained, model_name)()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Hold the ESM model OUTSIDE nn.Module's registry so it is excluded from
        # state_dict / .parameters() / optimizer. We move/cast it manually.
        self._esm = [model]
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.padding_idx = alphabet.padding_idx

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self._torch_dtype = {
            "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
        }[dtype]
        self._device = torch.device("cpu")
        self._ready = False

    # ---- device / dtype management (manual, since ESM isn't a submodule) -- #
    @property
    def model(self):
        return self._esm[0]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, device: torch.device | str) -> "ESM2Backbone":
        """Move and cast the (frozen) ESM model to ``device``. Call once on setup."""
        self._device = torch.device(device)
        m = self.model.to(self._device)
        if self._torch_dtype != torch.float32 and self._device.type == "cuda":
            m = m.to(self._torch_dtype)
        self._esm[0] = m
        self._ready = True
        return self

    # ---- core forward --------------------------------------------------- #
    @torch.no_grad()
    def _forward_tokens(self, seqs: List[str]) -> Tuple[torch.Tensor, List[int]]:
        """Run ESM on a list of sequences that each fit in one context window.

        Returns (reps (n, Lmax+2, D) float32, lengths) with BOS/EOS still present.
        """
        data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self._device)
        out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer].float()
        lengths = [len(s) for s in seqs]
        return reps, lengths

    @torch.no_grad()
    def _embed_short_batch(self, seqs: List[str]) -> List[torch.Tensor]:
        reps, lengths = self._forward_tokens(seqs)
        # residue i is at token position i+1 (BOS at 0)
        return [reps[j, 1 : 1 + L] for j, L in enumerate(lengths)]

    @torch.no_grad()
    def _embed_one_long(self, seq: str) -> torch.Tensor:
        L = len(seq)
        D = self._embed_dim
        acc = torch.zeros(L, D, device=self._device)
        cnt = torch.zeros(L, 1, device=self._device)
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < L:
            end = min(start + self.chunk_size, L)
            window = seq[start:end]
            rep = self._embed_short_batch([window])[0]  # (end-start, D)
            acc[start:end] += rep
            cnt[start:end] += 1.0
            if end == L:
                break
            start += step
        cnt = cnt.clamp_min(1.0)
        return acc / cnt

    @torch.no_grad()
    def embed(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of sequences.

        Returns
        -------
        emb  : (B, Lmax, D) float32, zero-padded
        mask : (B, Lmax) bool, True for real residues
        """
        if not self._ready:
            self.prepare(self._device)

        reps: List[torch.Tensor] = [None] * len(sequences)  # type: ignore[list-item]

        short_idx = [i for i, s in enumerate(sequences) if len(s) <= self.chunk_size]
        long_idx = [i for i, s in enumerate(sequences) if len(s) > self.chunk_size]

        if short_idx:
            short_reps = self._embed_short_batch([sequences[i] for i in short_idx])
            for i, r in zip(short_idx, short_reps):
                reps[i] = r
        for i in long_idx:
            reps[i] = self._embed_one_long(sequences[i])

        B = len(sequences)
        Lmax = max((r.shape[0] for r in reps), default=1)
        D = self._embed_dim
        emb = torch.zeros(B, Lmax, D, device=self._device)
        mask = torch.zeros(B, Lmax, dtype=torch.bool, device=self._device)
        for i, r in enumerate(reps):
            Li = r.shape[0]
            emb[i, :Li] = r
            mask[i, :Li] = True
        return emb, mask

    # ---- keep ESM weights out of checkpoints --------------------------- #
    def state_dict(self, *args, **kwargs):  # noqa: D401 - frozen, nothing to save
        return {}

    def _load_from_state_dict(self, *args, **kwargs):  # noqa: D401 - nothing to load
        return
