# ---------------------------------------------------------------------------
# Model: ESM-2 encoder + Transformer decoder
# ---------------------------------------------------------------------------

import math
import re
from contextlib import nullcontext

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ESM2TextToTextTransformer(nn.Module):
    """Encoder-decoder model that uses pretrained ESM-2 representations as memory."""

    def __init__(
        self,
        tgt_vocab_size,
        src_tokenizer=None,
        esm_model_name="esm2_t33_650M_UR50D",
        esm_repr_layer=None,
        freeze_esm=True,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_tgt_len=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tokenizer = src_tokenizer
        self.esm_model_name = esm_model_name
        self.freeze_esm = freeze_esm

        self.esm_encoder, self.esm_alphabet = self._load_esm2(esm_model_name)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_repr_layer = esm_repr_layer or self._infer_repr_layer(esm_model_name, self.esm_encoder)

        esm_dim = getattr(self.esm_encoder, "embed_dim", None)
        if esm_dim is None:
            raise ValueError("Could not infer ESM-2 embedding dimension from the loaded model.")

        if freeze_esm:
            self.esm_encoder.eval()
            for param in self.esm_encoder.parameters():
                param.requires_grad = False

        self.encoder_projection = nn.Linear(esm_dim, d_model)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder_dropout = nn.Dropout(dropout)

        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        # +2 for SOS/EOS tokens added by the dataset
        self.pos_enc = PositionalEncoding(d_model, max_len=max_tgt_len + 2, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_decoder_weights()

    def _load_esm2(self, esm_model_name):
        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM-2 requires the 'esm' package. Install it with 'pip install fair-esm'."
            ) from exc

        loader = getattr(esm.pretrained, esm_model_name, None)
        if loader is None:
            raise ValueError(f"Unknown ESM-2 model '{esm_model_name}' in esm.pretrained.")
        return loader()

    def _infer_repr_layer(self, esm_model_name, esm_encoder):
        match = re.search(r"_t(\d+)_", esm_model_name)
        if match is not None:
            return int(match.group(1))
        return getattr(esm_encoder, "num_layers", None)

    def _init_decoder_weights(self):
        modules = [
            self.encoder_projection,
            self.tgt_embed,
            self.decoder,
            self.fc_out,
        ]
        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_esm:
            self.esm_encoder.eval()
        return self

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def _src_to_sequences(self, src):
        if isinstance(src, (list, tuple)) and all(isinstance(item, str) for item in src):
            return list(src)

        if not torch.is_tensor(src):
            raise TypeError("src must be either a tensor of token ids or a list of protein sequences.")

        if self.src_tokenizer is None:
            raise ValueError("src_tokenizer is required when src is a tensor of token ids.")

        return [self.src_tokenizer.decode(row.detach().cpu(), strip_special=True) for row in src]

    def _encode_src(self, src, src_key_padding_mask=None):
        sequences = self._src_to_sequences(src)
        device = next(self.esm_encoder.parameters()).device
        _, _, esm_tokens = self.esm_batch_converter([(str(i), seq) for i, seq in enumerate(sequences)])
        esm_tokens = esm_tokens.to(device)

        grad_context = torch.no_grad() if self.freeze_esm else nullcontext()
        with grad_context:
            esm_out = self.esm_encoder(
                esm_tokens,
                repr_layers=[self.esm_repr_layer],
                return_contacts=False,
            )
            memory = esm_out["representations"][self.esm_repr_layer]

        if src_key_padding_mask is not None:
            memory = self._match_src_length(memory, src_key_padding_mask.size(1))

        memory = memory.to(self.encoder_projection.weight.device)
        memory = self.encoder_projection(memory)
        memory = self.encoder_norm(memory)
        return self.encoder_dropout(memory)

    def _match_src_length(self, memory, target_len):
        current_len = memory.size(1)
        if current_len == target_len:
            return memory
        if current_len > target_len:
            return memory[:, :target_len, :]

        pad_len = target_len - current_len
        pad = memory.new_zeros(memory.size(0), pad_len, memory.size(2))
        return torch.cat([memory, pad], dim=1)

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self._encode_src(src, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = self.pos_enc(self.tgt_embed(tgt_in))

        if tgt_mask is None:
            tgt_len = tgt_in.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt_in.device)

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)
