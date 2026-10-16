"""ESM-2 encoder + learned domain queries (DETR / Mask2Former style).

Each query predicts one domain: a mask over residues, a CATH label (C, A, T, H) and a confidence score.
Queries are matched to the true domains with the Hungarian algorithm, so domain order does not matter.
"""

from __future__ import annotations

import math
import re
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from residue_mlm.labels import AA_ALPHABET, PAD_ID, SPECIAL_VOCAB_SIZE

CATH_LEVELS = ("cath_c", "cath_a", "cath_t", "cath_h")


def load_esm2(esm_model_name: str):
    try:
        import esm
    except ImportError as exc:
        raise ImportError("ESM-2 requires the 'esm' package. Install it with 'pip install fair-esm'.") from exc

    loader = getattr(esm.pretrained, esm_model_name, None)
    if loader is None:
        raise ValueError(f"Unknown ESM-2 model '{esm_model_name}' in esm.pretrained.")
    return loader()


class DomainQueryModel(nn.Module):
    def __init__(
        self,
        *,
        cath_vocab_sizes: dict[str, int],
        esm_model_name: str | None = "esm2_t33_650M_UR50D",
        freeze_esm: bool = True,
        max_len: int = 1022,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_esm = freeze_esm
        self.esm = None

        if esm_model_name:
            # Residue embeddings from pretrained ESM-2
            self.esm, alphabet = load_esm2(esm_model_name)
            self.esm_batch_converter = alphabet.get_batch_converter()
            self.esm_repr_layer = int(re.search(r"_t(\d+)_", esm_model_name).group(1))
            self.input_proj = nn.Linear(self.esm.embed_dim, d_model)
            if freeze_esm:
                for param in self.esm.parameters():
                    param.requires_grad = False
        else:
            # Small transformer trained from scratch (no ESM)
            self.aa_embed = nn.Embedding(SPECIAL_VOCAB_SIZE + len(AA_ALPHABET), d_model, padding_idx=PAD_ID)
            self.position_embed = nn.Embedding(max_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.residue_norm = nn.LayerNorm(d_model)

        # Domain queries: self-attention between queries + cross-attention to residues
        self.queries = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.mask_query_proj = nn.Linear(d_model, d_model)
        self.mask_residue_proj = nn.Linear(d_model, d_model)
        self.score_head = nn.Linear(d_model, 1)
        self.cath_heads = nn.ModuleDict({level: nn.Linear(d_model, size) for level, size in cath_vocab_sizes.items()})

    def train(self, mode: bool = True):
        super().train(mode)
        if self.esm is not None and self.freeze_esm:
            self.esm.eval()
        return self

    def encode_residues(self, sequences: list[str], aa_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.esm is not None:
            _, _, esm_tokens = self.esm_batch_converter([(str(i), seq) for i, seq in enumerate(sequences)])
            esm_tokens = esm_tokens.to(aa_ids.device)
            with torch.no_grad() if self.freeze_esm else nullcontext():
                esm_out = self.esm(esm_tokens, repr_layers=[self.esm_repr_layer])
            # drop <cls> at position 0; positions after seq_len are padding and get masked
            hidden = esm_out["representations"][self.esm_repr_layer][:, 1 : aa_ids.size(1) + 1]
            hidden = self.input_proj(hidden)
        else:
            positions = torch.arange(aa_ids.size(1), device=aa_ids.device).unsqueeze(0)
            hidden = self.aa_embed(aa_ids) + self.position_embed(positions)
            hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        return self.residue_norm(hidden)

    def forward(self, sequences: list[str], aa_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        residues = self.encode_residues(sequences, aa_ids, attention_mask)  # B x L x d

        queries = self.queries.weight.unsqueeze(0).expand(aa_ids.size(0), -1, -1)  # B x Q x d
        queries = self.decoder(queries, residues, memory_key_padding_mask=~attention_mask)

        mask_logits = torch.einsum(
            "bqd,bld->bql", self.mask_query_proj(queries), self.mask_residue_proj(residues)
        ) / math.sqrt(self.d_model)  # B x Q x L

        outputs = {
            "mask_logits": mask_logits,
            "score_logits": self.score_head(queries).squeeze(-1),  # B x Q
        }
        for level_name, head in self.cath_heads.items():
            outputs[f"{level_name}_logits"] = head(queries)  # B x Q x vocab
        return outputs

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        domain_masks: list[torch.Tensor],
        domain_cath: list[torch.Tensor],
        seq_lens: torch.Tensor,
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = loss_weights or {"mask": 1.0, "score": 1.0, "cath": 0.5}
        mask_losses, score_losses = [], []
        cath_logits = {level: [] for level in CATH_LEVELS}
        cath_targets = {level: [] for level in CATH_LEVELS}

        for batch_idx, seq_len in enumerate(seq_lens.tolist()):
            logits = outputs["mask_logits"][batch_idx, :, :seq_len]  # Q x L
            gt_masks = domain_masks[batch_idx].to(logits.dtype)  # N x L
            score_target = torch.zeros_like(outputs["score_logits"][batch_idx])

            if gt_masks.size(0) > 0:
                query_idx, domain_idx = match_queries(logits, gt_masks)
                pred, target = logits[query_idx], gt_masks[domain_idx]
                mask_losses.append(F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target))

                # Confidence target = IoU between the predicted mask and its true domain
                pred_bool, target_bool = pred.detach() > 0, target > 0.5
                iou = (pred_bool & target_bool).sum(-1) / (pred_bool | target_bool).sum(-1).clamp(min=1)
                score_target[query_idx] = iou.to(score_target.dtype)

                for level_idx, level in enumerate(CATH_LEVELS):
                    cath_logits[level].append(outputs[f"{level}_logits"][batch_idx, query_idx])
                    cath_targets[level].append(domain_cath[batch_idx][domain_idx, level_idx])

            # Unmatched queries learn score 0 ("no domain")
            score_losses.append(F.binary_cross_entropy_with_logits(outputs["score_logits"][batch_idx], score_target))

        zero = outputs["mask_logits"].sum() * 0.0
        mask_loss = torch.stack(mask_losses).mean() if mask_losses else zero
        score_loss = torch.stack(score_losses).mean()
        total = weights["mask"] * mask_loss + weights["score"] * score_loss
        metrics = {"mask_loss": float(mask_loss.detach()), "score_loss": float(score_loss.detach())}

        for level in CATH_LEVELS:
            level_loss = outputs[f"{level}_logits"].sum() * 0.0  # keeps every head in the graph (DDP)
            if cath_targets[level]:
                logits_cat = torch.cat(cath_logits[level])
                targets_cat = torch.cat(cath_targets[level])
                if (targets_cat != -100).any():
                    level_loss = level_loss + F.cross_entropy(logits_cat, targets_cat, ignore_index=-100)
            total = total + weights["cath"] * level_loss
            metrics[f"{level}_loss"] = float(level_loss.detach())

        metrics["loss"] = float(total.detach())
        return total, metrics


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    dice = (2 * (probs * targets).sum(-1) + 1) / (probs.sum(-1) + targets.sum(-1) + 1)
    return (1 - dice).mean()


@torch.no_grad()
def match_queries(logits: torch.Tensor, gt_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hungarian matching of Q predicted masks to N true domains with cost = BCE + Dice."""
    logits, gt_masks = logits.float(), gt_masks.float()
    seq_len = logits.size(1)

    # Pairwise BCE (Q x N): softplus(x) - x * y, averaged over residues
    bce = (F.softplus(logits).sum(-1, keepdim=True) - logits @ gt_masks.T) / seq_len

    # Pairwise Dice (Q x N)
    probs = logits.sigmoid()
    dice = 1 - (2 * probs @ gt_masks.T + 1) / (probs.sum(-1, keepdim=True) + gt_masks.sum(-1).unsqueeze(0) + 1)

    query_idx, domain_idx = linear_sum_assignment((bce + dice).cpu().numpy())
    device = logits.device
    return torch.as_tensor(query_idx, device=device), torch.as_tensor(domain_idx, device=device)
