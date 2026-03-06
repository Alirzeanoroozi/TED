"""
Load a trained TED model, run inference on a sample, and plot attention weights
(cross-attention: decoder attending to encoder).
"""

import os
import sys
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

# Allow importing from parent when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_ import _load_paths
from tokenizer_ import TextTokenizer
from model import TextToTextTransformer


class AttentionCapture:
    """Captures attention weights from decoder cross-attention via forward hooks."""

    def __init__(self):
        self.weights = []  # List of (layer_idx, attn_weights)

    def __call__(self, module, module_in, module_out):
        # module_out is (attn_output, attn_output_weights) from MultiheadAttention
        attn_output, attn_weights = module_out
        if attn_weights is not None:
            self.weights.append(attn_weights.detach().cpu())

    def clear(self):
        self.weights.clear()


def load_model_and_data(checkpoint_path, data_path, device):
    """Load checkpoint, build model, load weights, and prepare tokenizers + sample."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    data_folder = data_path or ckpt_args.get("data_parquet_folder") or "data"
    if os.path.isdir(data_folder):
        paths = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".parquet") or f.endswith(".csv")
        ]
    else:
        paths = [data_folder] if os.path.exists(data_folder) else []
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(
            f"No .parquet or .csv files in {data_folder}. "
            "Set --data_path to a folder or file, e.g. data/ or data/chains.ted.with_seq.csv"
        )

    df = _load_paths(paths[:5])
    src_tokenizer = TextTokenizer().fit(df["sequence"].astype(str).tolist())
    tgt_tokenizer = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())

    src_vocab_size = ckpt["src_vocab_size"]
    tgt_vocab_size = ckpt["tgt_vocab_size"]

    d_model = ckpt_args.get("d_model", 256)
    nhead = ckpt_args.get("nhead", 8)
    num_encoder_layers = ckpt_args.get("num_encoder_layers", 4)
    num_decoder_layers = ckpt_args.get("num_decoder_layers", 4)
    dim_feedforward = ckpt_args.get("dim_feedforward", 1024)
    dropout = ckpt_args.get("dropout", 0.1)
    max_src_len = ckpt_args.get("max_src_len", 1024)
    max_tgt_len = ckpt_args.get("max_tgt_len", 128)

    model = TextToTextTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "df": df,
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
    }


def _patch_multihead_attn_for_weights(module):
    """Force need_weights=True and average_attn_weights=False for per-head weights."""
    _orig = module.forward

    def _forward(*args, **kwargs):
        kwargs["need_weights"] = True  # TransformerDecoderLayer passes False by default
        kwargs["average_attn_weights"] = False  # get per-head weights (N, n_heads, L, S)
        return _orig(*args, **kwargs)

    module.forward = _forward


def extract_attention(model, src_tensor, tgt_tensor, src_pad_id, tgt_pad_id, device):
    """
    Run forward pass and capture cross-attention weights from all decoder layers.
    Returns list of tensors [n_layers], each shape (batch, n_heads, tgt_len, src_len).
    """
    capture = AttentionCapture()
    hooks = []

    for i, layer in enumerate(model.decoder.layers):
        # TransformerDecoderLayer passes need_weights=False; patch to get weights
        _patch_multihead_attn_for_weights(layer.multihead_attn)
        h = layer.multihead_attn.register_forward_hook(capture)
        hooks.append(h)

    src_key_padding_mask = src_tensor.eq(src_pad_id)
    tgt_key_padding_mask = tgt_tensor.eq(tgt_pad_id)
    tgt_len = tgt_tensor.size(1)
    tgt_mask = model.generate_square_subsequent_mask(tgt_len, device)

    with torch.no_grad():
        _ = model(
            src_tensor.to(device),
            tgt_tensor.to(device),
            src_key_padding_mask=src_key_padding_mask.to(device),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask.to(device),
        )

    for h in hooks:
        h.remove()

    return capture.weights


def plot_attention(
    attn_weights_list,
    src_tokens,
    tgt_tokens,
    out_path="attention_weights.png",
    layer_idx=0,
    head_idx=None,
):
    """
    Plot attention heatmap.
    attn_weights_list: list of (n_heads, tgt_len, src_len) per layer
    """
    if layer_idx >= len(attn_weights_list):
        layer_idx = 0
    attn = attn_weights_list[layer_idx]
    # attn: (1, n_heads, tgt_len, src_len)
    attn = attn[0]
    n_heads, tgt_len, src_len = attn.shape

    if head_idx is not None:
        # Plot single head
        to_plot = attn[head_idx].numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(to_plot, aspect="auto", cmap="viridis")
        ax.set_xlabel("Source position")
        ax.set_ylabel("Target position")
        ax.set_title(f"Cross-attention (layer {layer_idx}, head {head_idx})")
        ax.set_xticks(range(min(src_len, len(src_tokens))))
        ax.set_xticklabels(src_tokens[:src_len], fontsize=6, rotation=90)
        ax.set_yticks(range(min(tgt_len, len(tgt_tokens))))
        ax.set_yticklabels(tgt_tokens[:tgt_len], fontsize=6)
        plt.colorbar(im, ax=ax)
    else:
        # Plot all heads in a grid
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for h in range(n_heads):
            r, c = h // n_cols, h % n_cols
            im = axes[r, c].imshow(attn[h].numpy(), aspect="auto", cmap="viridis")
            axes[r, c].set_title(f"Head {h}")
            axes[r, c].set_xlabel("Source")
            axes[r, c].set_ylabel("Target")
        for h in range(n_heads, n_rows * n_cols):
            r, c = h // n_cols, h % n_cols
            axes[r, c].axis("off")
        fig.suptitle(f"Cross-attention weights (layer {layer_idx})", fontsize=12)
        plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load TED model, run on a sample, plot attention weights."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="transformer_checkpoint.pt",
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Folder with parquet/csv or path to single file. Default: data/ or from checkpoint.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Row index of sample to use.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Decoder layer to plot (0-indexed).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Specific attention head to plot. If None, plot all heads.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_weights.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_model_and_data(args.checkpoint, args.data_path, device)
    model = loaded["model"]
    df = loaded["df"]
    src_tokenizer = loaded["src_tokenizer"]
    tgt_tokenizer = loaded["tgt_tokenizer"]
    max_src_len = loaded["max_src_len"]
    max_tgt_len = loaded["max_tgt_len"]

    sample_idx = min(args.sample_idx, len(df) - 1)
    row = df.iloc[sample_idx]
    src_text = str(row["sequence"]).strip()[:max_src_len]
    tgt_text = str(row["chopping_star"]).strip()[:max_tgt_len]

    src_ids = src_tokenizer.encode(src_text, add_sos_eos=True)
    tgt_ids = tgt_tokenizer.encode(tgt_text, add_sos_eos=True)
    tgt_in_ids = tgt_ids[:-1]

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
    tgt_tensor = torch.tensor(tgt_in_ids, dtype=torch.long).unsqueeze(0)

    src_tokens = [src_tokenizer.id2token.get(i, "?") for i in src_ids]
    tgt_tokens = [tgt_tokenizer.id2token.get(i, "?") for i in tgt_in_ids]

    print("Extracting attention weights...")
    attn_weights_list = extract_attention(
        model,
        src_tensor,
        tgt_tensor,
        src_tokenizer.pad_id,
        tgt_tokenizer.pad_id,
        device,
    )

    if not attn_weights_list:
        raise RuntimeError(
            "No attention weights captured. Ensure the model decoder uses MultiheadAttention "
            "and the patch for need_weights=True is applied correctly."
        )
    print(f"Captured {len(attn_weights_list)} layers. Plotting layer {args.layer}...")
    plot_attention(
        attn_weights_list,
        src_tokens,
        tgt_tokens,
        out_path=args.output,
        layer_idx=args.layer,
        head_idx=args.head,
    )


if __name__ == "__main__":
    main()
