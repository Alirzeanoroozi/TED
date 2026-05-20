"""Load a trained two-stage checkpoint and predict ``chopping_star`` strings.

Used by the benchmark runner and for ad-hoc inference. The frozen ESM2 backbone
is reloaded from pretrained weights (not stored in the checkpoint); only the
segmentation + CATH heads are restored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from two_stage.cath_vocab import CathVocab
from two_stage.model import TwoStageDomainModel


def load_two_stage(checkpoint_path: str | Path, device: torch.device | str = "cpu"):
    """Rebuild the model from a ``.pt`` saved by train.py and load head weights."""
    device = torch.device(device)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    args = ckpt.get("args", {})
    vocab = CathVocab.from_dict(ckpt["cath_vocab"])

    model = TwoStageDomainModel(
        cath_vocab=vocab,
        esm_model_name=args.get("esm_model_name", "esm2_t33_650M_UR50D"),
        d_model=args.get("d_model", 512),
        nhead=args.get("nhead", 8),
        num_seg_layers=args.get("num_seg_layers", 4),
        dim_feedforward=args.get("dim_feedforward", 2048),
        dropout=0.0,  # eval
        pair_dim=args.get("pair_dim", 128),
        cath_hidden=args.get("cath_hidden", 512),
        cath_cond_dim=args.get("cath_cond_dim", 64),
        esm_chunk_size=args.get("esm_chunk_size", 1022),
        esm_chunk_overlap=args.get("esm_chunk_overlap", 256),
        esm_dtype=args.get("esm_dtype", "bf16"),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Backbone keys are intentionally absent; warn only about head mismatches.
    head_missing = [k for k in missing if not k.startswith("backbone.")]
    if head_missing or unexpected:
        print(f"[warn] load_state_dict missing={head_missing} unexpected={unexpected}")
    model.to(device)
    model.prepare_backbone(device)
    model.eval()
    return model, args


@torch.no_grad()
def predict_chopping_star_batch(
    model: TwoStageDomainModel,
    sequences: List[str],
    *,
    batch_size: int = 4,
    cluster_method: str = "connected_components",
    pair_threshold: float = 0.5,
    domain_threshold: float = 0.5,
    min_domain_len: int = 20,
    classify_cath: bool = True,
    cath_decode: str = "greedy",
    cath_beam_width: int = 4,
    max_len: int | None = None,
) -> List[str]:
    preds: List[str] = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        if max_len is not None:
            chunk = [s[:max_len] for s in chunk]
        preds.extend(
            model.predict_chopping_star(
                chunk,
                cluster_method=cluster_method,
                pair_threshold=pair_threshold,
                domain_threshold=domain_threshold,
                min_domain_len=min_domain_len,
                classify_cath=classify_cath,
                cath_decode=cath_decode,
                cath_beam_width=cath_beam_width,
            )
        )
    return preds


def main():
    p = argparse.ArgumentParser(description="Predict chopping_star strings with a two-stage checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence", action="append", default=[], help="Raw AA sequence (repeatable).")
    p.add_argument("--device", default=None)
    p.add_argument("--min_domain_len", type=int, default=20)
    p.add_argument("--cluster_method", default="connected_components",
                   choices=["connected_components", "spectral"])
    p.add_argument("--cath_decode", default="greedy", choices=["greedy", "beam"])
    p.add_argument("--cath_beam_width", type=int, default=4)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_two_stage(args.checkpoint, device)
    if not args.sequence:
        p.error("provide at least one --sequence")
    seqs = args.sequence
    preds = predict_chopping_star_batch(
        model, seqs, min_domain_len=args.min_domain_len, cluster_method=args.cluster_method,
        cath_decode=args.cath_decode, cath_beam_width=args.cath_beam_width,
    )
    for s, pr in zip(seqs, preds):
        print(f"len={len(s)} -> {pr}")


if __name__ == "__main__":
    main()
