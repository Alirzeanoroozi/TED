"""TED-style domain segmentation evaluation pipeline.

Canonical internal indexing convention in this module:
- 0-based, half-open residue intervals: [start, end)

Expected legacy CSV annotation format for `label` / `predicted` columns:
- Domains separated by `*`
- Each domain chunk formatted as: `SEGMENTS | CLASS`
- Discontinuous segments separated by `_`
- Segment ranges formatted as `start-end` (1-based inclusive by default)

Example:
    23-142 | 1.20.120 * 148-262 | 3.30.460.10
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # package import path
    from .domain_boundry_dist import (
        boundary_distance_score,
        get_true_boundary_res,
        pred_domains_to_bounds,
    )
    from .ndo import ndo_score
    from .utils import convert_domain_dict_strings
except ImportError:  # direct script path
    from domain_boundry_dist import boundary_distance_score, get_true_boundary_res, pred_domains_to_bounds
    from ndo import ndo_score
    from utils import convert_domain_dict_strings


RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


@dataclass
class Domain:
    """One parsed domain with residue segments and an optional class label."""

    segments: List[Tuple[int, int]]
    class_label: Optional[str] = None


@dataclass
class ParsedAnnotation:
    """Parsed domain annotation plus parse diagnostics."""

    domains: List[Domain] = field(default_factory=list)
    parse_ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Configuration for per-target and batch evaluation."""

    iou_threshold: float = 0.8
    boundary_tolerance: int = 20
    input_indexing: str = "one_based_inclusive"
    include_deepdom_boundary_metrics: bool = False


def load_raw_ted_predictions(_: Any) -> pd.DataFrame:
    """Stub for future raw-output adapters.

    This evaluator intentionally focuses on canonical CSV/domain-annotation inputs.
    Implementations can add adapters here for raw AMPSampler/TED outputs once a
    stable raw schema is available.
    """

    raise NotImplementedError(
        "Raw TED/AMPSampler output adapter is not implemented. "
        "Use canonical CSV input with `label` and `predicted` columns."
    )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_annotation(annotation: Any, *, input_indexing: str = "one_based_inclusive") -> ParsedAnnotation:
    """Parse a legacy annotation string to canonical 0-based half-open segments.

    Parameters
    ----------
    annotation:
        Raw annotation string or NaN/None.
    input_indexing:
        - "one_based_inclusive" (default): input `start-end` is 1-based inclusive
        - "zero_based_half_open": input `start-end` is 0-based half-open
    """

    parsed = ParsedAnnotation()

    if _is_missing(annotation):
        parsed.warnings.append("annotation_missing")
        return parsed

    text = str(annotation).strip()
    if not text:
        parsed.warnings.append("annotation_empty")
        return parsed

    chunks = text.split("*")
    for chunk_idx, raw_chunk in enumerate(chunks):
        chunk = raw_chunk.strip()
        if not chunk:
            parsed.warnings.append(f"empty_chunk_at_index_{chunk_idx}")
            continue

        if "|" not in chunk:
            parsed.parse_ok = False
            parsed.errors.append(f"missing_pipe_in_chunk_{chunk_idx}:{chunk}")
            continue

        bounds_text, class_text = chunk.split("|", 1)
        bounds_text = bounds_text.strip()
        class_text = class_text.strip()
        class_label = None if class_text in {"", "-"} else class_text

        if not bounds_text:
            parsed.parse_ok = False
            parsed.errors.append(f"empty_bounds_in_chunk_{chunk_idx}")
            continue

        segments: List[Tuple[int, int]] = []
        for seg_idx, seg_raw in enumerate(bounds_text.split("_")):
            seg = seg_raw.strip()
            if not seg:
                parsed.parse_ok = False
                parsed.errors.append(f"empty_segment_chunk_{chunk_idx}_seg_{seg_idx}")
                continue

            m = RANGE_RE.match(seg)
            if not m:
                if seg.isdigit():
                    start_in = int(seg)
                    end_in = int(seg)
                else:
                    parsed.parse_ok = False
                    parsed.errors.append(f"invalid_segment_chunk_{chunk_idx}_seg_{seg_idx}:{seg}")
                    continue
            else:
                start_in, end_in = map(int, m.groups())

            if start_in > end_in:
                parsed.parse_ok = False
                parsed.errors.append(
                    f"descending_segment_chunk_{chunk_idx}_seg_{seg_idx}:{start_in}-{end_in}"
                )
                continue

            if input_indexing == "one_based_inclusive":
                if start_in < 1:
                    parsed.parse_ok = False
                    parsed.errors.append(
                        f"invalid_one_based_start_chunk_{chunk_idx}_seg_{seg_idx}:{start_in}"
                    )
                    continue
                start = start_in - 1
                end = end_in
            elif input_indexing == "zero_based_half_open":
                if start_in < 0:
                    parsed.parse_ok = False
                    parsed.errors.append(
                        f"invalid_zero_based_start_chunk_{chunk_idx}_seg_{seg_idx}:{start_in}"
                    )
                    continue
                start = start_in
                end = end_in
            else:
                raise ValueError(f"Unsupported input_indexing='{input_indexing}'")

            if end <= start:
                parsed.parse_ok = False
                parsed.errors.append(
                    f"non_positive_length_segment_chunk_{chunk_idx}_seg_{seg_idx}:{seg}"
                )
                continue

            segments.append((start, end))

        if segments:
            segments.sort(key=lambda x: (x[0], x[1]))
            parsed.domains.append(Domain(segments=segments, class_label=class_label))
        else:
            parsed.parse_ok = False
            parsed.errors.append(f"no_valid_segments_in_chunk_{chunk_idx}")

    return parsed


def _domain_residue_sets(parsed: ParsedAnnotation) -> List[set[int]]:
    residue_sets: List[set[int]] = []
    for domain in parsed.domains:
        residues: set[int] = set()
        for start, end in domain.segments:
            residues.update(range(start, end))
        residue_sets.append(residues)
    return residue_sets


def _max_end(parsed: ParsedAnnotation) -> int:
    max_end = 0
    for domain in parsed.domains:
        for _, end in domain.segments:
            max_end = max(max_end, end)
    return max_end


def _infer_nres(
    row: pd.Series,
    gt_parsed: ParsedAnnotation,
    pred_parsed: ParsedAnnotation,
    *,
    nres_col: str,
    seq_col: str,
) -> Tuple[int, str]:
    if nres_col in row and not _is_missing(row[nres_col]):
        try:
            nres_val = int(row[nres_col])
            if nres_val > 0:
                return nres_val, nres_col
        except (TypeError, ValueError):
            pass

    if seq_col in row and not _is_missing(row[seq_col]):
        seq = str(row[seq_col]).strip()
        if seq:
            return len(seq), seq_col

    inferred = max(_max_end(gt_parsed), _max_end(pred_parsed))
    return inferred, "parsed_ranges"


def _build_domain_dict(parsed: ParsedAnnotation, nres: int) -> Tuple[Dict[str, List[int]], List[str]]:
    """Build domain dict expected by `ndo.py` and `domain_boundry_dist.py`.

    Returns
    -------
    domain_dict, warnings
    """

    warnings: List[str] = []
    domain_dict: Dict[str, List[int]] = {}

    assigned: set[int] = set()
    for idx, domain in enumerate(parsed.domains, start=1):
        residues: set[int] = set()
        for start, end in domain.segments:
            if start >= nres:
                warnings.append(f"domain_{idx}_outside_chain_start_{start}_nres_{nres}")
                continue
            clipped_end = min(end, nres)
            if clipped_end <= start:
                continue
            residues.update(range(start, clipped_end))

        clean_residues = sorted(r for r in residues if 0 <= r < nres and r not in assigned)
        dropped = len(residues) - len(clean_residues)
        if dropped:
            warnings.append(f"domain_{idx}_dropped_overlap_or_oob_residues_{dropped}")

        if clean_residues:
            key = f"D{idx}"
            domain_dict[key] = clean_residues
            assigned.update(clean_residues)

    domain_dict["linker"] = sorted(set(range(nres)).difference(assigned))
    return domain_dict, warnings


def _pairwise_iou(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)


def _greedy_assignment(iou_matrix: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    pairs: List[Tuple[int, int]] = []
    used_row: set[int] = set()
    used_col: set[int] = set()

    ranked: List[Tuple[float, int, int]] = []
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            ranked.append((float(iou_matrix[i, j]), i, j))
    ranked.sort(key=lambda x: x[0], reverse=True)

    total = 0.0
    for score, i, j in ranked:
        if i in used_row or j in used_col:
            continue
        used_row.add(i)
        used_col.add(j)
        pairs.append((i, j))
        total += score
    return total, pairs


def _optimal_assignment(iou_matrix: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Maximize sum of pairwise IoU under one-to-one matching."""

    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0 or n_gt == 0:
        return 0.0, []

    if max(n_pred, n_gt) > 12:
        return _greedy_assignment(iou_matrix)

    if n_pred <= n_gt:

        @lru_cache(maxsize=None)
        def solve(i: int, used_mask: int) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
            if i == n_pred:
                return 0.0, tuple()

            best_score = -1.0
            best_pairs: Tuple[Tuple[int, int], ...] = tuple()
            for j in range(n_gt):
                if used_mask & (1 << j):
                    continue
                sub_score, sub_pairs = solve(i + 1, used_mask | (1 << j))
                cand_score = float(iou_matrix[i, j]) + sub_score
                if cand_score > best_score:
                    best_score = cand_score
                    best_pairs = ((i, j),) + sub_pairs
            return best_score, best_pairs

        score, pairs = solve(0, 0)
        return float(score), list(pairs)

    @lru_cache(maxsize=None)
    def solve(j: int, used_mask: int) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
        if j == n_gt:
            return 0.0, tuple()

        best_score = -1.0
        best_pairs: Tuple[Tuple[int, int], ...] = tuple()
        for i in range(n_pred):
            if used_mask & (1 << i):
                continue
            sub_score, sub_pairs = solve(j + 1, used_mask | (1 << i))
            cand_score = float(iou_matrix[i, j]) + sub_score
            if cand_score > best_score:
                best_score = cand_score
                best_pairs = ((i, j),) + sub_pairs
        return best_score, best_pairs

    score, pairs = solve(0, 0)
    return float(score), list(pairs)


def chain_overlap_metrics(
    gt_parsed: ParsedAnnotation,
    pred_parsed: ParsedAnnotation,
    *,
    iou_threshold: float,
) -> Dict[str, float]:
    """Compute chain-level IoU and correctly-parsed proportion."""

    gt_sets = _domain_residue_sets(gt_parsed)
    pred_sets = _domain_residue_sets(pred_parsed)

    n_gt = len(gt_sets)
    n_pred = len(pred_sets)
    denom = max(n_gt, n_pred)

    if denom == 0:
        return {
            "iou_chain": 1.0,
            "correct_prop": 1.0,
            "correct_cath_prop": 1.0,
        }

    if n_gt == 0 or n_pred == 0:
        return {
            "iou_chain": 0.0,
            "correct_prop": 0.0,
            "correct_cath_prop": 0.0,
        }

    iou_matrix = np.zeros((n_pred, n_gt), dtype=float)
    for i, pset in enumerate(pred_sets):
        for j, gset in enumerate(gt_sets):
            iou_matrix[i, j] = _pairwise_iou(pset, gset)

    total_iou, pairs = _optimal_assignment(iou_matrix)

    correct = 0
    correct_cath = 0
    for i, j in pairs:
        iou_val = float(iou_matrix[i, j])
        if iou_val >= iou_threshold:
            correct += 1

        pred_label = pred_parsed.domains[i].class_label
        gt_label = gt_parsed.domains[j].class_label
        if pred_label == gt_label:
            correct_cath += 1

    return {
        "iou_chain": total_iou / denom,
        "correct_prop": correct / denom,
        "correct_cath_prop": correct_cath / denom,
    }


def _compute_boundary_classification_metrics(
    true_bounds: Sequence[int],
    pred_bounds: Sequence[int],
    *,
    nres: int,
    tolerance: int,
) -> Dict[str, float]:
    """DeepDom-style boundary P/R/MCC from segmentation-derived boundaries.

    Residue-level positives are defined as residues within +/- `tolerance`
    from any boundary residue.
    """

    if nres <= 0:
        return {
            "boundary_precision": math.nan,
            "boundary_recall": math.nan,
            "boundary_mcc": math.nan,
        }

    true_mask = np.zeros(nres, dtype=bool)
    pred_mask = np.zeros(nres, dtype=bool)

    def _mark(mask: np.ndarray, boundaries: Sequence[int]) -> None:
        for b in boundaries:
            # boundary extractors may include nres as a terminal boundary marker
            b_clipped = min(max(int(b), 0), nres - 1)
            lo = max(0, b_clipped - tolerance)
            hi = min(nres, b_clipped + tolerance + 1)
            mask[lo:hi] = True

    _mark(true_mask, true_bounds)
    _mark(pred_mask, pred_bounds)

    tp = int(np.logical_and(true_mask, pred_mask).sum())
    fp = int(np.logical_and(~true_mask, pred_mask).sum())
    fn = int(np.logical_and(true_mask, ~pred_mask).sum())
    tn = int(np.logical_and(~true_mask, ~pred_mask).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_mcc": mcc,
    }


def evaluate_target(
    label: Any,
    predicted: Any,
    *,
    nres: Optional[int] = None,
    sequence: Optional[str] = None,
    config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """Evaluate one target from label/predicted domain annotation strings."""

    cfg = config or EvalConfig()

    gt_parsed = parse_annotation(label, input_indexing=cfg.input_indexing)
    pred_parsed = parse_annotation(predicted, input_indexing=cfg.input_indexing)

    row_like = pd.Series({"nres": nres, "input": sequence})
    inferred_nres, nres_source = _infer_nres(
        row_like,
        gt_parsed,
        pred_parsed,
        nres_col="nres",
        seq_col="input",
    )

    gt_dict, gt_dict_warnings = _build_domain_dict(gt_parsed, inferred_nres)
    pred_dict, pred_dict_warnings = _build_domain_dict(pred_parsed, inferred_nres)

    out: Dict[str, Any] = {
        "nres": inferred_nres,
        "nres_source": nres_source,
        "gt_parse_ok": gt_parsed.parse_ok,
        "pred_parse_ok": pred_parsed.parse_ok,
        "gt_parse_errors": ";".join(gt_parsed.errors),
        "pred_parse_errors": ";".join(pred_parsed.errors),
        "gt_parse_warnings": ";".join(gt_parsed.warnings + gt_dict_warnings),
        "pred_parse_warnings": ";".join(pred_parsed.warnings + pred_dict_warnings),
        "gt_domain_count": len([k for k in gt_dict if k != "linker"]),
        "pred_domain_count": len([k for k in pred_dict if k != "linker"]),
    }

    out["domain_count_match"] = int(out["gt_domain_count"] == out["pred_domain_count"])

    overlap = chain_overlap_metrics(gt_parsed, pred_parsed, iou_threshold=cfg.iou_threshold)
    out.update(overlap)

    # Backward-compatible alias used in existing CSVs.
    out["correct_cath"] = out["correct_cath_prop"]

    try:
        out["ndo"] = float(ndo_score(gt_dict, pred_dict))
    except Exception as exc:  # pragma: no cover - safety net around legacy code
        out["ndo"] = math.nan
        out["ndo_error"] = str(exc)

    try:
        # If both have no non-linker domains, define as 1.0 by convention.
        if out["gt_domain_count"] == 0 and out["pred_domain_count"] == 0:
            out["boundary_distance_score"] = 1.0
        else:
            true_boundaries = get_true_boundary_res(gt_dict)
            out["boundary_distance_score"] = float(boundary_distance_score(pred_dict, true_boundaries))
    except Exception as exc:  # pragma: no cover - safety net around legacy code
        out["boundary_distance_score"] = math.nan
        out["boundary_distance_error"] = str(exc)

    if cfg.include_deepdom_boundary_metrics:
        try:
            true_bounds = pred_domains_to_bounds(gt_dict)
            pred_bounds = pred_domains_to_bounds(pred_dict)
            out.update(
                _compute_boundary_classification_metrics(
                    true_bounds,
                    pred_bounds,
                    nres=inferred_nres,
                    tolerance=cfg.boundary_tolerance,
                )
            )
        except Exception as exc:  # pragma: no cover - safety net around legacy code
            out["boundary_precision"] = math.nan
            out["boundary_recall"] = math.nan
            out["boundary_mcc"] = math.nan
            out["boundary_classification_error"] = str(exc)

    # Normalized serializations using existing utils logic.
    gt_names, gt_bounds = convert_domain_dict_strings(gt_dict)
    pred_names, pred_bounds = convert_domain_dict_strings(pred_dict)
    out["gt_domain_names_norm"] = gt_names
    out["gt_domain_bounds_norm"] = gt_bounds
    out["pred_domain_names_norm"] = pred_names
    out["pred_domain_bounds_norm"] = pred_bounds

    out["gt_domain_type"] = "single" if out["gt_domain_count"] == 1 else "multi" if out["gt_domain_count"] > 1 else "zero"

    return out


def _pick_target_id(row: pd.Series, idx: int, id_col: Optional[str]) -> str:
    if id_col and id_col in row and not _is_missing(row[id_col]):
        return str(row[id_col])
    for candidate in ("target_id", "id"):
        if candidate in row and not _is_missing(row[candidate]):
            return str(row[candidate])
    return str(idx)


def evaluate_dataframe(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    pred_col: str = "predicted",
    seq_col: str = "input",
    nres_col: str = "nres",
    id_col: Optional[str] = None,
    config: Optional[EvalConfig] = None,
) -> pd.DataFrame:
    """Batch-evaluate a dataframe and return one scored row per input row."""

    cfg = config or EvalConfig()

    missing = [c for c in (label_col, pred_col) if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        target_id = _pick_target_id(row, idx, id_col)

        nres_value: Optional[int] = None
        if nres_col in df.columns and not _is_missing(row.get(nres_col)):
            try:
                nres_value = int(row[nres_col])
            except (TypeError, ValueError):
                nres_value = None

        sequence_value: Optional[str] = None
        if seq_col in df.columns and not _is_missing(row.get(seq_col)):
            sequence_value = str(row[seq_col])

        metrics = evaluate_target(
            row[label_col],
            row[pred_col],
            nres=nres_value,
            sequence=sequence_value,
            config=cfg,
        )
        metrics["target_id"] = target_id
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    return metrics_df


def summarize_metrics(metrics_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Produce summary dict and a flat summary dataframe."""

    metric_cols = [
        "iou_chain",
        "correct_prop",
        "correct_cath_prop",
        "ndo",
        "boundary_distance_score",
    ]
    for opt_col in ("boundary_precision", "boundary_recall", "boundary_mcc"):
        if opt_col in metrics_df.columns:
            metric_cols.append(opt_col)

    summary_rows: List[Dict[str, Any]] = []

    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return float(value)

    def _make_group(group_name: str, group_df: pd.DataFrame) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "group": group_name,
            "n": int(len(group_df)),
            "gt_parse_ok_rate": _optional_float(group_df["gt_parse_ok"].mean()) if len(group_df) else None,
            "pred_parse_ok_rate": _optional_float(group_df["pred_parse_ok"].mean()) if len(group_df) else None,
            "domain_count_match_rate": _optional_float(group_df["domain_count_match"].mean()) if len(group_df) else None,
        }
        for col in metric_cols:
            if col in group_df.columns:
                row[f"mean_{col}"] = _optional_float(group_df[col].mean(skipna=True))
                row[f"median_{col}"] = _optional_float(group_df[col].median(skipna=True))
        return row

    overall = _make_group("overall", metrics_df)
    summary_rows.append(overall)

    if "gt_domain_type" in metrics_df.columns:
        for domain_type in ["zero", "single", "multi"]:
            gdf = metrics_df[metrics_df["gt_domain_type"] == domain_type]
            summary_rows.append(_make_group(f"gt_{domain_type}", gdf))

    summary_df = pd.DataFrame(summary_rows)

    summary_dict: Dict[str, Any] = {
        "n_targets": int(len(metrics_df)),
        "groups": summary_rows,
    }

    return summary_dict, summary_df


def evaluate_csv(
    input_csv: Path,
    output_dir: Path,
    *,
    label_col: str = "label",
    pred_col: str = "predicted",
    seq_col: str = "input",
    nres_col: str = "nres",
    id_col: Optional[str] = None,
    config: Optional[EvalConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """End-to-end CSV evaluation and output file writing."""

    df = pd.read_csv(input_csv)
    metrics_df = evaluate_dataframe(
        df,
        label_col=label_col,
        pred_col=pred_col,
        seq_col=seq_col,
        nres_col=nres_col,
        id_col=id_col,
        config=config,
    )

    metrics_for_output = metrics_df.copy()
    rename_map: Dict[str, str] = {}
    for col in metrics_for_output.columns:
        if col in df.columns:
            rename_map[col] = f"{col}_eval"
    if rename_map:
        metrics_for_output = metrics_for_output.rename(columns=rename_map)

    per_target = pd.concat([df.reset_index(drop=True), metrics_for_output.reset_index(drop=True)], axis=1)
    summary_dict, summary_df = summarize_metrics(metrics_df)
    cfg = config or EvalConfig()
    summary_dict["config"] = {
        "iou_threshold": cfg.iou_threshold,
        "boundary_tolerance": cfg.boundary_tolerance,
        "input_indexing": cfg.input_indexing,
        "include_deepdom_boundary_metrics": cfg.include_deepdom_boundary_metrics,
    }
    summary_dict["input"] = str(input_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    per_target_path = output_dir / "per_target_metrics.csv"
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"

    per_target.to_csv(per_target_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_dict, fh, indent=2)

    return per_target, summary_dict, summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TED-style domain segmentation predictions.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--label-col", default="label", help="Ground-truth annotation column.")
    parser.add_argument("--pred-col", default="predicted", help="Prediction annotation column.")
    parser.add_argument("--seq-col", default="input", help="Sequence column used to infer nres.")
    parser.add_argument("--nres-col", default="nres", help="Residue-count column.")
    parser.add_argument("--id-col", default=None, help="Optional target ID column.")
    parser.add_argument(
        "--input-indexing",
        default="one_based_inclusive",
        choices=["one_based_inclusive", "zero_based_half_open"],
        help="Indexing convention of input annotation ranges.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.8,
        help="IoU threshold for correctly parsed proportion.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=20,
        help="Tolerance (residues) for DeepDom-style boundary P/R/MCC.",
    )
    parser.add_argument(
        "--deepdom-boundary-metrics",
        action="store_true",
        help="Enable DeepDom-style boundary precision/recall/MCC from segmentation boundaries.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = EvalConfig(
        iou_threshold=args.iou_threshold,
        boundary_tolerance=args.boundary_tolerance,
        input_indexing=args.input_indexing,
        include_deepdom_boundary_metrics=args.deepdom_boundary_metrics,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    evaluate_csv(
        input_path,
        output_dir,
        label_col=args.label_col,
        pred_col=args.pred_col,
        seq_col=args.seq_col,
        nres_col=args.nres_col,
        id_col=args.id_col,
        config=cfg,
    )

    print(f"Wrote: {output_dir / 'per_target_metrics.csv'}")
    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
