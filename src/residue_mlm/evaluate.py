"""Standalone evaluation: sequence → boundaries + CATH (chopping_star)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from residue_mlm.config_utils import load_config
from residue_mlm.dataset import ResidueMLMDataset, build_level_vocabularies, collate_residue_batch, split_dataframe
from residue_mlm.decode import batch_predictions_to_chopping_stars
from residue_mlm.labels import SPECIAL_VOCAB_SIZE, AA_ALPHABET
from residue_mlm.model import ResidueMLM
from residue_mlm.predict import evaluate_sequence_predictions, predict_residue_labels
from data._dataset import load_data, resolve_parquet_inputs
from residue_mlm.dataset import _target_column


def load_model_from_checkpoint(checkpoint_path: str, cath_vocabs, args) -> ResidueMLM:
    cath_vocab_sizes = {name: vocab.vocab_size for name, vocab in cath_vocabs.items()}
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate residue MLM sequence-only predictions.")
    parser.add_argument("--config", default="configs/residue_mlm_config.yaml")
    parser.add_argument("--checkpoint", default="artifacts/residue_mlm_checkpoint.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    args_cli = parser.parse_args()

    args = load_config(args_cli.config)
    parquet_files = resolve_parquet_inputs(args.data_parquet_folder)
    df = load_data(parquet_files)
    target_column = _target_column(df)
    cath_vocabs = build_level_vocabularies(df, target_column)
    _, val_df, test_df = split_dataframe(df, val_size=args.val_size, test_size=args.test_size)
    eval_df = val_df if args_cli.split == "val" else test_df

    dataset = ResidueMLMDataset(
        eval_df,
        cath_vocabs,
        max_len=args.max_len,
        random_crop=False,
        target_column=target_column,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_residue_batch(batch, max_len=args.max_len),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args_cli.checkpoint, cath_vocabs, args).to(device)
    model.eval()

    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if args_cli.max_batches and batch_idx >= args_cli.max_batches:
            break
        tensor_batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        metrics = evaluate_sequence_predictions(
            model,
            tensor_batch,
            cath_vocabs,
            batch["chopping_stars"],
            batch["left_offsets"],
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

        if batch_idx == 0:
            preds = predict_residue_labels(model, tensor_batch)
            pred_stars = batch_predictions_to_chopping_stars(
                preds["domain_ids"],
                preds,
                tensor_batch["seq_len"],
                batch["left_offsets"],
                cath_vocabs,
            )
            print("Example GT:  ", batch["chopping_stars"][0])
            print("Example PRED:", pred_stars[0])

    if count:
        print("\nMean metrics:")
        for key, value in sorted(totals.items()):
            print(f"  {key}: {value / count:.4f}")


if __name__ == "__main__":
    main()
