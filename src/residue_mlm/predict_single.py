"""Predict domain boundaries and CATH labels for one protein sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from residue_mlm.config_utils import load_config
from residue_mlm.decode import residue_predictions_to_chopping_star
from residue_mlm.labels import AA_TO_ID, LevelVocabulary, PAD_ID, SPECIAL_VOCAB_SIZE, AA_ALPHABET
from residue_mlm.merge_windows import iter_window_offsets, merge_window_predictions
from residue_mlm.model import ResidueMLM
from residue_mlm.predict import predict_residue_labels


def vocabs_from_checkpoint(cath_vocabs_dict: Dict[str, Dict[str, int]]) -> Dict[str, LevelVocabulary]:
    vocabs: Dict[str, LevelVocabulary] = {}
    for name, token2id in cath_vocabs_dict.items():
        vocab = LevelVocabulary(name)
        vocab.token2id = dict(token2id)
        vocab.id2token = {idx: tok for tok, idx in token2id.items()}
        vocabs[name] = vocab
    return vocabs


def load_model_and_vocabs(
    checkpoint_path: str,
    config_path: str,
) -> tuple[ResidueMLM, Dict[str, LevelVocabulary], argparse.Namespace]:
    args = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "cath_vocabs" in checkpoint:
        cath_vocabs = vocabs_from_checkpoint(checkpoint["cath_vocabs"])
        cath_vocab_sizes = {name: vocab.vocab_size for name, vocab in cath_vocabs.items()}
    else:
        cath_vocab_sizes = checkpoint["cath_vocab_sizes"]
        cath_vocabs = {name: LevelVocabulary(name) for name in cath_vocab_sizes}

    model = ResidueMLM(
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        aa_vocab_size=SPECIAL_VOCAB_SIZE + len(AA_ALPHABET),
        max_num_domains=args.max_num_domains,
        max_domain_index=args.max_domain_index,
        cath_vocab_sizes=cath_vocab_sizes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, cath_vocabs, args


def sequence_to_batch(
    sequence: str,
    *,
    max_len: int = 128,
    left_offset: int = 0,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq = sequence.strip().upper()
    full_len = len(seq)
    left_offset = max(0, min(left_offset, max(0, full_len - 1)))
    window = seq[left_offset : left_offset + max_len]
    nres = len(window)

    aa_ids = [AA_TO_ID.get(aa, 1) for aa in window]
    positions = list(range(nres))

    batch = {
        "aa_ids": torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device),
        "positions": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "domain_ids": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "num_domains": torch.zeros(1, dtype=torch.long, device=device),
        "cath_c": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "cath_a": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "cath_t": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "cath_h": torch.zeros(1, max_len, dtype=torch.long, device=device),
        "seq_len": torch.tensor([nres], dtype=torch.long, device=device),
    }
    batch["aa_ids"][0, :nres] = torch.tensor(aa_ids, dtype=torch.long, device=device)
    batch["positions"][0, :nres] = torch.tensor(positions, dtype=torch.long, device=device)
    return batch


def windows_to_batch(
    sequence: str,
    offsets: List[int],
    *,
    max_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Stack multiple windows from one sequence into a single batch."""
    batch_size = len(offsets)
    seq = sequence.strip().upper()

    batch = {
        "aa_ids": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long, device=device),
        "positions": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "domain_ids": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "num_domains": torch.zeros(batch_size, dtype=torch.long, device=device),
        "cath_c": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "cath_a": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "cath_t": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "cath_h": torch.zeros(batch_size, max_len, dtype=torch.long, device=device),
        "seq_len": torch.zeros(batch_size, dtype=torch.long, device=device),
    }

    for batch_idx, left_offset in enumerate(offsets):
        window = seq[left_offset : left_offset + max_len]
        nres = len(window)
        batch["seq_len"][batch_idx] = nres
        batch["aa_ids"][batch_idx, :nres] = torch.tensor(
            [AA_TO_ID.get(aa, 1) for aa in window], dtype=torch.long, device=device
        )
        batch["positions"][batch_idx, :nres] = torch.arange(nres, device=device)
    return batch


def _window_preds_from_batch(
    preds: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    offsets: List[int],
) -> List[Tuple[int, Dict[str, List[int]]]]:
    window_preds: List[Tuple[int, Dict[str, List[int]]]] = []
    for batch_idx, left_offset in enumerate(offsets):
        length = int(batch["seq_len"][batch_idx])
        window_preds.append(
            (
                left_offset,
                {
                    "domain_ids": preds["domain_ids"][batch_idx, :length].cpu().tolist(),
                    "cath_c": preds["cath_c"][batch_idx, :length].cpu().tolist(),
                    "cath_a": preds["cath_a"][batch_idx, :length].cpu().tolist(),
                    "cath_t": preds["cath_t"][batch_idx, :length].cpu().tolist(),
                    "cath_h": preds["cath_h"][batch_idx, :length].cpu().tolist(),
                },
            )
        )
    return window_preds


@torch.no_grad()
def predict_one_window(
    sequence: str,
    model: ResidueMLM,
    cath_vocabs: Dict[str, LevelVocabulary],
    *,
    max_len: int = 128,
    left_offset: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Predict a single window (legacy behaviour for len <= max_len)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch = sequence_to_batch(sequence, max_len=max_len, left_offset=left_offset, device=device)
    preds = predict_residue_labels(model, batch)
    length = int(batch["seq_len"][0])

    chopping_star = residue_predictions_to_chopping_star(
        preds["domain_ids"][0, :length].tolist(),
        preds["cath_c"][0, :length].tolist(),
        preds["cath_a"][0, :length].tolist(),
        preds["cath_t"][0, :length].tolist(),
        preds["cath_h"][0, :length].tolist(),
        cath_vocabs,
        left_offset=left_offset,
        index_base=1,
    )

    return {
        "sequence": sequence.strip().upper(),
        "full_len": len(sequence.strip()),
        "merged": False,
        "num_windows": 1,
        "window_offsets": [left_offset],
        "sequence_window": sequence.strip().upper()[left_offset : left_offset + length],
        "left_offset": left_offset,
        "seq_len": length,
        "chopping_star": chopping_star,
        "domain_ids": preds["domain_ids"][0, :length].cpu().tolist(),
        "cath_c": preds["cath_c"][0, :length].cpu().tolist(),
        "cath_a": preds["cath_a"][0, :length].cpu().tolist(),
        "cath_t": preds["cath_t"][0, :length].cpu().tolist(),
        "cath_h": preds["cath_h"][0, :length].cpu().tolist(),
    }


@torch.no_grad()
def predict_one_sequence(
    sequence: str,
    model: ResidueMLM,
    cath_vocabs: Dict[str, LevelVocabulary],
    *,
    max_len: int = 128,
    stride: Optional[int] = None,
    left_offset: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Predict full chain; automatically merges windows when len(sequence) > max_len."""
    seq = sequence.strip().upper()
    full_len = len(seq)

    if left_offset is not None:
        return predict_one_window(
            sequence, model, cath_vocabs, max_len=max_len, left_offset=left_offset, device=device
        )

    if full_len <= max_len:
        return predict_one_window(sequence, model, cath_vocabs, max_len=max_len, left_offset=0, device=device)

    window_stride = stride if stride is not None else max_len // 2
    offsets = iter_window_offsets(full_len, max_len, window_stride)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    window_preds: List[Tuple[int, Dict[str, List[int]]]] = []
    for chunk_start in range(0, len(offsets), 8):
        chunk_offsets = offsets[chunk_start : chunk_start + 8]
        batch = windows_to_batch(seq, chunk_offsets, max_len=max_len, device=device)
        preds = predict_residue_labels(model, batch)
        window_preds.extend(_window_preds_from_batch(preds, batch, chunk_offsets))

    merged = merge_window_predictions(full_len, window_preds)
    chopping_star = residue_predictions_to_chopping_star(
        merged["domain_ids"],
        merged["cath_c"],
        merged["cath_a"],
        merged["cath_t"],
        merged["cath_h"],
        cath_vocabs,
        left_offset=0,
        index_base=1,
    )

    return {
        "sequence": seq,
        "full_len": full_len,
        "merged": True,
        "num_windows": len(offsets),
        "window_offsets": offsets,
        "stride": window_stride,
        "seq_len": full_len,
        "chopping_star": chopping_star,
        "domain_ids": merged["domain_ids"],
        "cath_c": merged["cath_c"],
        "cath_a": merged["cath_a"],
        "cath_t": merged["cath_t"],
        "cath_h": merged["cath_h"],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict boundaries + CATH for one sequence.")
    parser.add_argument("--sequence", required=True, help="Amino-acid sequence string")
    parser.add_argument("--config", default="configs/residue_mlm_config.yaml")
    parser.add_argument("--checkpoint", default="artifacts/residue_mlm_checkpoint.pt")
    parser.add_argument(
        "--left-offset",
        type=int,
        default=None,
        help="Predict only this window (0-based). Disables automatic full-chain merge.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride for long sequences (default: max_len/2)",
    )
    parser.add_argument("--json", action="store_true", help="Print full output as JSON")
    args_cli = parser.parse_args()

    repo_root = SRC_ROOT.parent
    config_path = Path(args_cli.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path

    model, cath_vocabs, args = load_model_and_vocabs(str(checkpoint_path), str(config_path))
    result = predict_one_sequence(
        args_cli.sequence,
        model,
        cath_vocabs,
        max_len=args.max_len,
        stride=args_cli.stride,
        left_offset=args_cli.left_offset,
    )

    if args_cli.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Input length: {result['full_len']}")
    if result.get("merged"):
        print(f"Merged {result['num_windows']} windows (stride={result['stride']}, max_len={args.max_len})")
        print(f"Window starts: {result['window_offsets']}")
    else:
        start = result.get("left_offset", 0) + 1
        end = result.get("left_offset", 0) + result["seq_len"]
        print(f"Single window: residues {start}–{end}")
    print(f"\nPredicted chopping_star:\n{result['chopping_star']}")


if __name__ == "__main__":
    main()
