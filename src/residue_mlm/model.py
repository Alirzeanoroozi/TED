"""Residue-level BERT-style MLM with exact positional embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from residue_mlm.labels import AA_ALPHABET, PAD_ID, SPECIAL_VOCAB_SIZE


class ResidueMLM(nn.Module):
    """Masked LM over residues using sequence, exact position, domain count, and CATH levels."""

    def __init__(
        self,
        *,
        max_len: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        aa_vocab_size: int = SPECIAL_VOCAB_SIZE + len(AA_ALPHABET),
        max_num_domains: int = 32,
        max_domain_index: int = 32,
        cath_vocab_sizes: dict[str, int],
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pad_id = PAD_ID

        self.aa_embed = nn.Embedding(aa_vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embed = nn.Embedding(max_len, d_model)
        self.num_domains_embed = nn.Embedding(max_num_domains + 1, d_model)
        self.domain_id_embed = nn.Embedding(max_domain_index + 1, d_model, padding_idx=0)

        self.cath_embeds = nn.ModuleDict(
            {
                level: nn.Embedding(size, d_model, padding_idx=PAD_ID)
                for level, size in cath_vocab_sizes.items()
            }
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(d_model)

        self.aa_head = nn.Linear(d_model, aa_vocab_size)
        self.domain_head = nn.Linear(d_model, max_domain_index + 1)
        self.cath_heads = nn.ModuleDict(
            {level: nn.Linear(d_model, size) for level, size in cath_vocab_sizes.items()}
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def _combine_inputs(
        self,
        aa_ids,
        positions,
        num_domains,
        domain_ids,
        cath_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.aa_embed(aa_ids)
        hidden = hidden + self.position_embed(positions)
        hidden = hidden + self.domain_id_embed(domain_ids)

        domain_count = self.num_domains_embed(
            num_domains.clamp(max=self.num_domains_embed.num_embeddings - 1)
        )
        hidden = hidden + domain_count.unsqueeze(1)

        for level_name, level_ids in cath_inputs.items():
            hidden = hidden + self.cath_embeds[level_name](level_ids)

        return self.input_norm(hidden)

    def forward(
        self,
        aa_ids_input: torch.Tensor,
        positions: torch.Tensor,
        num_domains: torch.Tensor,
        domain_ids_input: torch.Tensor,
        cath_inputs: dict[str, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self._combine_inputs(
            aa_ids_input,
            positions,
            num_domains,
            domain_ids_input,
            cath_inputs,
        )

        if attention_mask is None:
            attention_mask = positions.new_ones(positions.shape, dtype=torch.bool)

        key_padding_mask = ~attention_mask
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)

        outputs = {
            "aa_logits": self.aa_head(encoded),
            "domain_logits": self.domain_head(encoded),
        }
        for level_name, head in self.cath_heads.items():
            outputs[f"{level_name}_logits"] = head(encoded)
        return outputs

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = loss_weights or {
            "aa": 1.0,
            "domain": 0.25,
            "cath_c": 0.5,
            "cath_a": 0.5,
            "cath_t": 0.5,
            "cath_h": 0.5,
        }
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total = logits["aa_logits"].new_zeros(())
        metrics: dict[str, float] = {}

        task_map = {
            "aa": ("aa_logits", "aa_labels", "aa"),
            "domain": ("domain_logits", "domain_labels", "domain"),
            "cath_c": ("cath_c_logits", "cath_c_labels", "cath_c"),
            "cath_a": ("cath_a_logits", "cath_a_labels", "cath_a"),
            "cath_t": ("cath_t_logits", "cath_t_labels", "cath_t"),
            "cath_h": ("cath_h_logits", "cath_h_labels", "cath_h"),
        }

        for task_name, (logit_key, label_key, weight_key) in task_map.items():
            if label_key not in labels:
                continue
            task_loss = criterion(
                logits[logit_key].reshape(-1, logits[logit_key].size(-1)),
                labels[label_key].reshape(-1),
            )
            weight = weights.get(weight_key, 1.0)
            total = total + weight * task_loss
            metrics[f"{task_name}_loss"] = float(task_loss.detach())

        metrics["loss"] = float(total.detach())
        return total, metrics
