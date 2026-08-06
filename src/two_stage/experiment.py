"""Frozen, group-aware data partitions for reproducible two-stage experiments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from two_stage.cath_vocab import CathVocab


MANIFEST_VERSION = 1
_AUGMENTED_SUFFIX_RE = re.compile(r"__aug_\d+$", flags=re.IGNORECASE)
_HOMOLOGY_GROUP_CANDIDATES = (
    "homology_cluster",
    "mmseqs_cluster",
    "sequence_cluster",
    "cluster_id",
)


def _present(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan", "<na>"}


def _stable_unit_interval(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _row_fingerprints(df: pd.DataFrame) -> list[str]:
    ids = df["uniprot_id"].tolist() if "uniprot_id" in df.columns else [""] * len(df)
    out = []
    for uid, seq, label in zip(ids, df["sequence"], df["chopping_star"]):
        text = f"{uid}\0{str(seq).strip()}\0{str(label).strip()}"
        out.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return out


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint the ordered rows used by ``_load_paths``."""
    h = hashlib.sha256()
    for row_hash in _row_fingerprints(df):
        h.update(row_hash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def infer_split_groups(
    df: pd.DataFrame, group_col: str = "auto"
) -> Tuple[pd.Series, str]:
    """Choose leakage-control groups, preferring explicit homology clusters.

    ``auto`` first uses a known homology-cluster column when one is populated.
    Otherwise original and augmented crops are grouped by ``augmented_from`` or
    the base UniProt id.  The latter prevents direct augmentation leakage but is
    not a substitute for a homology clustering column.
    """
    selected = None
    if group_col != "auto":
        if group_col not in df.columns:
            raise ValueError(
                f"split group column {group_col!r} is absent; columns={list(df.columns)}"
            )
        selected = group_col
    else:
        for candidate in _HOMOLOGY_GROUP_CANDIDATES:
            if candidate in df.columns and df[candidate].map(_present).any():
                selected = candidate
                break

    row_hashes = _row_fingerprints(df)
    augmented = (
        df["augmented_from"].tolist()
        if "augmented_from" in df.columns
        else [None] * len(df)
    )
    uniprot = (
        df["uniprot_id"].tolist()
        if "uniprot_id" in df.columns
        else [None] * len(df)
    )
    augmentation_groups: list[str] = []
    for source, uid, fallback in zip(augmented, uniprot, row_hashes):
        if _present(source):
            group = str(source).strip()
        elif _present(uid):
            group = _AUGMENTED_SUFFIX_RE.sub("", str(uid).strip())
        else:
            group = f"sequence:{fallback}"
        augmentation_groups.append(group)

    if selected is not None:
        groups = []
        for value, fallback_group in zip(df[selected].tolist(), augmentation_groups):
            groups.append(
                str(value).strip()
                if _present(value)
                else f"augmentation_fallback:{fallback_group}"
            )
        return pd.Series(groups, index=df.index, dtype="string"), selected

    return pd.Series(augmentation_groups, index=df.index, dtype="string"), "augmentation_source"


def build_split_manifest(
    df: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_col: str = "auto",
) -> pd.DataFrame:
    """Assign whole groups to train/validation/test using a stable hash."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to less than 1")
    groups, source = infer_split_groups(df, group_col=group_col)
    split_by_group = {}
    for group in groups.unique().tolist():
        u = _stable_unit_interval(seed, str(group))
        if u < test_ratio:
            split = "test"
        elif u < test_ratio + val_ratio:
            split = "val"
        else:
            split = "train"
        split_by_group[str(group)] = split

    split = groups.map(split_by_group)
    counts = split.value_counts().to_dict()
    required = [name for name, ratio in (("train", 1.0), ("val", val_ratio), ("test", test_ratio)) if ratio > 0]
    empty = [name for name in required if int(counts.get(name, 0)) == 0]
    if empty:
        raise ValueError(
            f"stable group split produced empty partitions {empty}; use more groups or another seed"
        )

    fingerprint = dataset_fingerprint(df)
    return pd.DataFrame(
        {
            "manifest_version": MANIFEST_VERSION,
            "dataset_fingerprint": fingerprint,
            "dataset_row": range(len(df)),
            "row_fingerprint": _row_fingerprints(df),
            "split_group": groups.astype(str),
            "split_group_source": source,
            "split_seed": int(seed),
            "split": split.astype(str),
        }
    )


def save_split_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def load_and_validate_split_manifest(
    df: pd.DataFrame, path: str | Path
) -> pd.DataFrame:
    path = Path(path)
    manifest = pd.read_csv(path)
    required = {
        "manifest_version",
        "dataset_fingerprint",
        "dataset_row",
        "row_fingerprint",
        "split_group",
        "split",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"split manifest {path} is missing columns: {sorted(missing)}")
    if len(manifest) != len(df):
        raise ValueError(
            f"split manifest has {len(manifest)} rows but current data has {len(df)}"
        )
    versions = set(manifest["manifest_version"].astype(int).tolist())
    if versions != {MANIFEST_VERSION}:
        raise ValueError(f"unsupported split manifest version(s): {sorted(versions)}")
    expected_dataset = dataset_fingerprint(df)
    recorded = set(manifest["dataset_fingerprint"].astype(str).tolist())
    if recorded != {expected_dataset}:
        raise ValueError(
            "split manifest was made for different data or file ordering; regenerate it explicitly"
        )
    ordered = manifest.sort_values("dataset_row").reset_index(drop=True)
    if ordered["dataset_row"].astype(int).tolist() != list(range(len(df))):
        raise ValueError("split manifest dataset_row must be exactly 0..N-1")
    if ordered["row_fingerprint"].astype(str).tolist() != _row_fingerprints(df):
        raise ValueError("split manifest row fingerprints do not match the current data")
    unknown = set(ordered["split"].astype(str)).difference({"train", "val", "test"})
    if unknown:
        raise ValueError(f"split manifest contains unknown split names: {sorted(unknown)}")
    return ordered


def split_dataframe(
    df: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a validated manifest without changing within-split row order."""
    splits = []
    for name in ("train", "val", "test"):
        positions = manifest.loc[manifest["split"] == name, "dataset_row"].astype(int)
        splits.append(df.iloc[positions.tolist()].reset_index(drop=True))
    return tuple(splits)  # type: ignore[return-value]


def write_experiment_metadata(
    manifest: pd.DataFrame,
    cath_vocab: CathVocab,
    path: str | Path,
) -> None:
    """Write a compact audit record next to a prepared manifest/vocabulary."""
    group_source = (
        str(manifest["split_group_source"].iloc[0])
        if "split_group_source" in manifest.columns and len(manifest)
        else "unknown"
    )
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_fingerprint": str(manifest["dataset_fingerprint"].iloc[0]),
        "split_group_source": group_source,
        "split_seed": int(manifest["split_seed"].iloc[0]),
        "rows_by_split": {
            str(k): int(v) for k, v in manifest["split"].value_counts().items()
        },
        "groups_by_split": {
            split: int(part["split_group"].nunique())
            for split, part in manifest.groupby("split")
        },
        "cath_vocab_level_sizes": cath_vocab.level_sizes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
