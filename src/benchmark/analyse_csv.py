"""Backward-compatible CSV analysis script.

This reads a CSV with `label` and `predicted`
columns, computing `iou_chain`, `correct_prop`, and `correct_cath`, and writing
`results_iou.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

try:  # package import path
    from .ted_eval import EvalConfig, evaluate_target
except ImportError:  # direct script path
    from ted_eval import EvalConfig, evaluate_target


def get_iou(predicted_chopped: str, actual_chopped: str) -> Tuple[float, float, float]:
    """Compute `(iou_chain, correct_prop, correct_cath)` for one prediction/label pair.

    Any parsing failure returns zeros, matching previous script behavior.
    """

    metrics = evaluate_target(
        actual_chopped,
        predicted_chopped,
        config=EvalConfig(iou_threshold=0.8, input_indexing="one_based_inclusive"),
    )

    if not metrics["gt_parse_ok"] or not metrics["pred_parse_ok"]:
        return 0.0, 0.0, 0.0

    return float(metrics["iou_chain"]), float(metrics["correct_prop"]), float(metrics["correct_cath"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute IoU-style metrics for a CSV.")
    parser.add_argument(
        "--input",
        default="wandb_export_2025-09-23T10_04_42.561+03_00.csv",
        help="Input CSV containing at least `label` and `predicted`.",
    )
    parser.add_argument("--output", default="results_iou.csv", help="Output CSV path.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"label", "predicted"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["iou_chain"], df["correct_prop"], df["correct_cath"] = zip(
        *df.apply(lambda row: get_iou(row["predicted"], row["label"]), axis=1)
    )

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(df["iou_chain"].mean())
    print(df["correct_prop"].mean())
    print(df["correct_cath"].mean())

    iou_chain_nonzero = df[df["iou_chain"] != 0]
    correct_prop_nonzero = df[df["correct_prop"] != 0]
    correct_cath_nonzero = df[df["correct_cath"] != 0]

    print(iou_chain_nonzero["iou_chain"].mean() if not iou_chain_nonzero.empty else 0)
    print(correct_prop_nonzero["correct_prop"].mean() if not correct_prop_nonzero.empty else 0)
    print(correct_cath_nonzero["correct_cath"].mean() if not correct_cath_nonzero.empty else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
