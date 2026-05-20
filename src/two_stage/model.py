"""Two-stage domain model: shared frozen ESM2 backbone + Stage A segmentation
head + Stage B hierarchical CATH classifier.

Stage A (segmentation)
    A small trainable Transformer trunk on top of frozen ESM2 embeddings, with
    two heads:
      * residue head      -> P(residue belongs to a domain vs linker/NDR)
      * pairwise head     -> P(residue i, j in the same domain)  [symmetric]
    At inference the pairwise matrix is clustered into domains (clustering.py),
    so the domain count is emergent rather than generated.

Stage B (CATH)
    Each domain's residue embeddings are masked-mean-pooled (ESM features +
    task-adapted trunk features) and classified by four hierarchical heads
    (C, A, T, H).  Each deeper head is conditioned on the chosen parent level
    (teacher-forced in training, argmax at inference), exploiting the hierarchy.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from two_stage.esm_backbone import ESM2Backbone
from two_stage.cath_vocab import CathVocab, N_LEVELS
from two_stage import clustering, serialize


# --------------------------------------------------------------------------- #
# Stage A: segmentation head
# --------------------------------------------------------------------------- #
class SegmentationHead(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pair_dim: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(esm_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.residue_head = nn.Linear(d_model, 1)
        self.q_proj = nn.Linear(d_model, pair_dim)
        self.k_proj = nn.Linear(d_model, pair_dim)
        self.pair_bias = nn.Parameter(torch.zeros(1))
        self.pair_scale = 1.0 / math.sqrt(pair_dim)
        self.d_model = d_model

    def forward(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """emb (B,L,esm_dim), mask (B,L) bool True=real.

        Returns residue_logit (B,L), pair_logit (B,L,L), trunk_rep (B,L,d_model).
        """
        key_padding_mask = ~mask  # True = ignore
        h = self.trunk(self.input_proj(emb), src_key_padding_mask=key_padding_mask)
        residue_logit = self.residue_head(h).squeeze(-1)

        q = self.q_proj(h)
        k = self.k_proj(h)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.pair_scale  # (B,L,L)
        pair_logit = 0.5 * (scores + scores.transpose(1, 2)) + self.pair_bias
        return residue_logit, pair_logit, h


# --------------------------------------------------------------------------- #
# Stage B: hierarchical CATH classifier
# --------------------------------------------------------------------------- #
class CathClassifier(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        level_sizes: Sequence[int],
        hidden: int = 512,
        cond_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(level_sizes) == N_LEVELS
        self.level_sizes = list(level_sizes)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList()
        self.cond_embeds = nn.ModuleList()
        in_dim = hidden
        for level in range(N_LEVELS):
            self.heads.append(nn.Linear(in_dim, level_sizes[level]))
            if level < N_LEVELS - 1:
                self.cond_embeds.append(nn.Embedding(level_sizes[level], cond_dim))
                in_dim = hidden + cond_dim

    def forward(
        self, feat: torch.Tensor, parent_ids: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """feat (N, feat_dim) one row per domain.

        parent_ids (N, 4) supplies teacher-forced parent ids for conditioning;
        if None, the model conditions on its own argmax (autoregressive over the
        hierarchy) -- used at inference.

        Returns (list of 4 logit tensors, pred_ids (N,4)).
        """
        h = self.trunk(feat)
        logits: List[torch.Tensor] = []
        pred_ids: List[torch.Tensor] = []

        logit1 = self.heads[0](h)
        logits.append(logit1)
        pred1 = logit1.argmax(dim=-1)
        pred_ids.append(pred1)

        prev_pred = pred1
        for level in range(1, N_LEVELS):
            if parent_ids is not None:
                cond_id = parent_ids[:, level - 1]
            else:
                cond_id = prev_pred
            cond = self.cond_embeds[level - 1](cond_id)
            logit = self.heads[level](torch.cat([h, cond], dim=-1))
            logits.append(logit)
            prev_pred = logit.argmax(dim=-1)
            pred_ids.append(prev_pred)

        return logits, torch.stack(pred_ids, dim=-1)


# --------------------------------------------------------------------------- #
# Combined model
# --------------------------------------------------------------------------- #
class TwoStageDomainModel(nn.Module):
    def __init__(
        self,
        cath_vocab: CathVocab,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        d_model: int = 512,
        nhead: int = 8,
        num_seg_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pair_dim: int = 128,
        cath_hidden: int = 512,
        cath_cond_dim: int = 64,
        esm_chunk_size: int = 1022,
        esm_chunk_overlap: int = 256,
        esm_dtype: str = "bf16",
    ):
        super().__init__()
        self.cath_vocab = cath_vocab
        self.backbone = ESM2Backbone(
            model_name=esm_model_name,
            chunk_size=esm_chunk_size,
            chunk_overlap=esm_chunk_overlap,
            dtype=esm_dtype,
        )
        esm_dim = self.backbone.embed_dim
        self.seg_head = SegmentationHead(
            esm_dim=esm_dim, d_model=d_model, nhead=nhead,
            num_layers=num_seg_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, pair_dim=pair_dim,
        )
        # CATH feature = pooled ESM embedding (PLM semantics) ++ pooled trunk rep.
        self.cath_feat_dim = esm_dim + d_model
        self.cath_head = CathClassifier(
            feat_dim=self.cath_feat_dim, level_sizes=cath_vocab.level_sizes,
            hidden=cath_hidden, cond_dim=cath_cond_dim, dropout=dropout,
        )
        self._init_trainable_weights()

    def _init_trainable_weights(self):
        for module in (self.seg_head, self.cath_head):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ---- helpers -------------------------------------------------------- #
    def prepare_backbone(self, device):
        self.backbone.prepare(device)

    def embed(self, sequences: List[str]):
        return self.backbone.embed(sequences)

    @staticmethod
    def _masked_pool(feats: torch.Tensor, res_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool (L, D) over a boolean residue mask (L,)."""
        if res_mask.sum() == 0:
            return feats.new_zeros(feats.shape[-1])
        return feats[res_mask].mean(dim=0)

    def _domain_features(
        self,
        emb: torch.Tensor,            # (B, L, esm_dim)
        trunk_rep: torch.Tensor,      # (B, L, d_model)
        domains: List[Tuple[int, torch.Tensor]],  # (sample_idx, res_mask (L,) bool)
    ) -> torch.Tensor:
        feats = []
        for sample_idx, res_mask in domains:
            e = self._masked_pool(emb[sample_idx], res_mask)
            t = self._masked_pool(trunk_rep[sample_idx], res_mask)
            feats.append(torch.cat([e, t], dim=-1))
        if not feats:
            return emb.new_zeros(0, self.cath_feat_dim)
        return torch.stack(feats, dim=0)

    # ---- training forward ---------------------------------------------- #
    def forward_train(
        self,
        emb: torch.Tensor,
        mask: torch.Tensor,
        domains: List[Tuple[int, torch.Tensor]],
        parent_ids: Optional[torch.Tensor],
    ):
        """Run both stages for a training step.

        ``domains`` are GROUND-TRUTH domains (teacher-forced segmentation) so the
        CATH head learns classification decoupled from segmentation errors.
        """
        residue_logit, pair_logit, trunk_rep = self.seg_head(emb, mask)
        if domains:
            feat = self._domain_features(emb, trunk_rep, domains)
            cath_logits, _ = self.cath_head(feat, parent_ids=parent_ids)
        else:
            cath_logits = None
        return {
            "residue_logit": residue_logit,
            "pair_logit": pair_logit,
            "trunk_rep": trunk_rep,
            "cath_logits": cath_logits,
        }

    # ---- inference ------------------------------------------------------ #
    @torch.no_grad()
    def predict_chopping_star(
        self,
        sequences: List[str],
        *,
        cluster_method: str = "connected_components",
        pair_threshold: float = 0.5,
        domain_threshold: float = 0.5,
        min_domain_len: int = 20,
        classify_cath: bool = True,
    ) -> List[str]:
        self.eval()
        emb, mask = self.embed(sequences)
        residue_logit, pair_logit, trunk_rep = self.seg_head(emb, mask)
        res_prob = torch.sigmoid(residue_logit)
        pair_prob = torch.sigmoid(pair_logit)

        outputs: List[str] = []
        for b, seq in enumerate(sequences):
            L = len(seq)
            in_dom = res_prob[b, :L].detach().cpu().numpy()
            pmat = pair_prob[b, :L, :L].detach().cpu().numpy()
            assign = clustering.comembership_to_domains(
                pmat, in_dom, method=cluster_method,
                pair_threshold=pair_threshold, domain_threshold=domain_threshold,
                min_domain_len=min_domain_len,
            )
            n_dom = int(assign.max())
            cath_codes: List[Optional[str]] = [None] * n_dom

            if classify_cath and n_dom > 0:
                domain_masks = []
                for k in range(1, n_dom + 1):
                    rm = torch.from_numpy(assign == k).to(emb.device)
                    # pad mask to L dim of emb
                    full = torch.zeros(emb.shape[1], dtype=torch.bool, device=emb.device)
                    full[:L] = rm
                    domain_masks.append((b, full))
                feat = self._domain_features(emb, trunk_rep, domain_masks)
                _, pred_ids = self.cath_head(feat, parent_ids=None)
                for k in range(n_dom):
                    cath_codes[k] = self.cath_vocab.decode(pred_ids[k].tolist())

            outputs.append(
                serialize.assignment_and_cath_to_chopping_star(assign, cath_codes)
            )
        return outputs
