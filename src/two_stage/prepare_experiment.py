"""Prepare one frozen split manifest and CATH vocabulary for all ablation arms.

This is a CPU-only metadata step.  It does not load ESM or train a model.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
for candidate in (str(SRC_DIR), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from data.dataset_ import _load_paths  # noqa: E402
from two_stage.cath_vocab import CathVocab  # noqa: E402
from two_stage.experiment import (  # noqa: E402
    build_split_manifest,
    load_and_validate_split_manifest,
    save_split_manifest,
    split_dataframe,
    write_experiment_metadata,
)


_CATH_RE = re.compile(r"\|\s*([\d.]+)")


def resolve_parquet_inputs(data_path: str) -> list[str]:
    path = Path(data_path).expanduser()
    path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        files = sorted(str(item) for item in path.glob("*.parquet"))
        if files:
            return files
        raise RuntimeError(f"No .parquet files under {path}")
    raise FileNotFoundError(f"Data path does not exist: {path}")


def extract_cath_codes(chopping_stars) -> list[str]:
    codes: list[str] = []
    for annotation in chopping_stars:
        codes.extend(_CATH_RE.findall(str(annotation)))
    return codes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze group-aware TED train/val/test splits and the train-only CATH vocab."
    )
    parser.add_argument("--data_parquet_folder", default="data/all_parquet")
    parser.add_argument(
        "--manifest_path",
        default="artifacts/two_stage_experiment/split_manifest.csv",
    )
    parser.add_argument(
        "--cath_vocab_path",
        default="artifacts/two_stage_experiment/cath_vocab.json",
    )
    parser.add_argument(
        "--metadata_path",
        default="artifacts/two_stage_experiment/metadata.json",
    )
    parser.add_argument("--split_group_col", default="auto")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--cath_min_count", type=int, default=5)
    parser.add_argument(
        "--require_homology_group",
        action="store_true",
        help="Fail if auto grouping can only prevent augmentation-family leakage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared manifest/vocabulary deliberately.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    vocab_path = Path(args.cath_vocab_path)
    metadata_path = Path(args.metadata_path)
    existing = [path for path in (manifest_path, vocab_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "prepared experiment files already exist; reuse them or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    df = _load_paths(resolve_parquet_inputs(args.data_parquet_folder)).reset_index(drop=True)
    manifest = build_split_manifest(
        df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        group_col=args.split_group_col,
    )
    source = str(manifest["split_group_source"].iloc[0])
    if args.require_homology_group and source == "augmentation_source":
        raise ValueError(
            "No populated homology-cluster column was found. Add one to the parquet data "
            "or pass its name with --split_group_col."
        )

    save_split_manifest(manifest, manifest_path)
    # Re-read and validate the serialized artifact, then build labels from train only.
    manifest = load_and_validate_split_manifest(df, manifest_path)
    train_df, _, _ = split_dataframe(df, manifest)
    codes = extract_cath_codes(train_df["chopping_star"].astype(str).tolist())
    vocab = CathVocab.build(codes, min_count=args.cath_min_count)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    write_experiment_metadata(manifest, vocab, metadata_path)

    print(f"Prepared split manifest: {manifest_path}")
    print(f"Prepared CATH vocabulary: {vocab_path}")
    print(f"Split group source: {source}")
    print(f"Rows by split: {manifest['split'].value_counts().to_dict()}")
    print(f"CATH vocabulary sizes: {vocab.level_sizes}")


if __name__ == "__main__":
    main()
