"""Benchmark the TWO-STAGE TEDPred model on the Chainsaw CATH-1363 test set.

Mirrors ``run_benchmark.py`` but loads the ESM2 two-stage checkpoint and predicts
``chopping_star`` strings with the segmentation + hierarchical-CATH heads.  All
evaluation, summary, and figure code is reused from ``run_benchmark`` so outputs
match the baseline format (predictions.csv, per_chain_metrics.csv, summary.json,
figure6a.png, figure_cath.png).

Usage (repo root, 'ted' env):
    python benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py \
        --checkpoint artifacts/two_stage_checkpoint.pt \
        --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv \
        --output-dir benchmark/chainsaw_cath1363_from_scratch/results_two_stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
for p in (str(THIS_DIR), str(SRC_DIR), str(BENCHMARK_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the baseline benchmark's evaluation / figure / summary helpers.
import run_benchmark as rb  # noqa: E402
from two_stage.infer import (  # noqa: E402
    load_two_stage,
    predict_chopping_star_with_aux_batch,
)
from two_stage.targets import parse_chopping_star, residue_assignment  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_fingerprint(bench_df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for _, row in bench_df.iterrows():
        payload = "\0".join(
            [
                str(row.get("target_id", "")),
                str(row.get("sequence", "")),
                str(row.get("label", "")),
                str(row.get("n_domains", "")),
            ]
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_metadata(
    bench_df: pd.DataFrame,
    checkpoint: Path,
    *,
    batch_size: int,
    min_domain_len: int,
    cluster_method: str,
    cath_decode: str,
    cath_beam_width: int,
    deletion_fusion_alpha: float,
    max_len: int,
) -> dict:
    return {
        "cache_schema": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "benchmark_fingerprint": _benchmark_fingerprint(bench_df),
        "n_targets": int(len(bench_df)),
        "batch_size": int(batch_size),
        "min_domain_len": int(min_domain_len),
        "cluster_method": str(cluster_method),
        "cath_decode": str(cath_decode),
        "cath_beam_width": int(cath_beam_width),
        "deletion_fusion_alpha": float(deletion_fusion_alpha),
        "max_len": int(max_len),
    }


def run_inference(
    bench_df: pd.DataFrame,
    model,
    *,
    batch_size: int,
    min_domain_len: int,
    cluster_method: str,
    cath_decode: str,
    cath_beam_width: int,
    deletion_fusion_alpha: float,
    max_len: int,
    cache_path: Path | None,
    cache_metadata: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    aux_path = cache_path.with_name(cache_path.stem + "_aux.parquet") if cache_path else None
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json") if cache_path else None
    if cache_path and cache_path.exists() and meta_path and meta_path.exists():
        recorded_metadata = json.loads(meta_path.read_text())
        cached = pd.read_csv(cache_path)
        ids_match = cached["target_id"].astype(str).tolist() == bench_df["target_id"].astype(str).tolist()
        metadata_match = cache_metadata is not None and recorded_metadata == cache_metadata
        aux_complete = aux_path is not None and aux_path.exists()
        if ids_match and metadata_match and aux_complete:
            print(f"All predictions cached at {cache_path}; skipping inference.")
            aux_cached = pd.read_parquet(aux_path)
            return cached, aux_cached
        print(f"Ignoring stale or incomplete prediction cache at {cache_path}.")

    target_ids = bench_df["target_id"].astype(str).tolist()
    sequences = bench_df["sequence"].astype(str).tolist()

    t0 = time.time()
    aux = predict_chopping_star_with_aux_batch(
        model,
        sequences,
        batch_size=batch_size,
        cluster_method=cluster_method,
        min_domain_len=min_domain_len,
        cath_decode=cath_decode,
        cath_beam_width=cath_beam_width,
        deletion_fusion_alpha=deletion_fusion_alpha,
        max_len=max_len,
    )
    dt = time.time() - t0
    print(f"Inference on {len(sequences)} chains in {dt:.1f}s ({len(sequences)/max(dt,1e-9):.1f} seq/s)")

    preds = [row["chopping_star"] for row in aux]
    pred_df = pd.DataFrame({"target_id": target_ids, "pred_chopping_star": preds})
    aux_df = pd.DataFrame(
        {
            "target_id": target_ids,
            "deletion_score": [row["deletion_score"].astype(float).tolist() for row in aux],
            "deletion_logit": [row["deletion_logit"].astype(float).tolist() for row in aux],
            "residue_prob": [row["residue_prob"].astype(float).tolist() for row in aux],
            "base_residue_prob": [row["base_residue_prob"].astype(float).tolist() for row in aux],
            "deletion_budget_fraction": [float(row["deletion_budget_fraction"]) for row in aux],
            "deletion_budget": [float(row["deletion_budget"]) for row in aux],
        }
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(cache_path, index=False)
        print(f"Saved predictions -> {cache_path}")
        aux_df.to_parquet(aux_path, index=False)
        print(f"Saved deletion aux scores -> {aux_path}")
        if meta_path is not None and cache_metadata is not None:
            meta_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n")
            print(f"Saved prediction provenance -> {meta_path}")
    return pred_df, aux_df


def run_inference_multi(bench_df, model, modes, *, batch_size, min_domain_len,
                        cluster_method, max_len):
    """Run several CATH-decode modes sharing one segmentation pass per batch.

    ``modes``: list of (label, cath_decode, beam_width). Returns {label: pred_df}.
    """
    target_ids = bench_df["target_id"].astype(str).tolist()
    sequences = [s[:max_len] for s in bench_df["sequence"].astype(str).tolist()]
    collected = {label: [] for (label, _, _) in modes}
    t0 = time.time()
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        res = model.predict_chopping_star_multi(
            chunk, modes, cluster_method=cluster_method, min_domain_len=min_domain_len,
        )
        for label in collected:
            collected[label].extend(res[label])
    print(f"Ablation inference on {len(sequences)} chains in {time.time()-t0:.1f}s "
          f"({len(modes)} CATH modes, shared segmentation)")
    return {label: pd.DataFrame({"target_id": target_ids, "pred_chopping_star": preds})
            for label, preds in collected.items()}


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(y_score.astype(float))
    pos_rank_sum = float(ranks[y].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score.astype(float))
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float(precision[y_sorted].sum() / n_pos)


def boundary_mask_from_assign(assign: np.ndarray, window: int = 2) -> np.ndarray:
    n = len(assign)
    out = np.zeros(n, dtype=bool)
    for i in range(max(0, n - 1)):
        if int(assign[i]) != int(assign[i + 1]):
            s = max(0, i - window)
            e = min(n, i + window + 2)
            out[s:e] = True
    return out


def topk_f1(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    y = y_true.astype(bool)
    if k is None:
        k = int(y.sum())
    if k <= 0 or y.sum() == 0:
        return float("nan")
    k = min(k, len(y))
    pred = np.zeros(len(y), dtype=bool)
    pred[np.argsort(-y_score.astype(float))[:k]] = True
    tp = float((pred & y).sum())
    precision = tp / max(1.0, float(pred.sum()))
    recall = tp / max(1.0, float(y.sum()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def add_deletion_metrics(
    bench_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    aux_df: pd.DataFrame | None,
    *,
    boundary_window: int = 2,
) -> pd.DataFrame:
    if aux_df is None or aux_df.empty:
        return metrics_df
    aux_by_id = {str(r["target_id"]): r for _, r in aux_df.iterrows()}
    rows = []
    for _, row in metrics_df.iterrows():
        tid = str(row["target_id"])
        bench_row = bench_df[bench_df["target_id"].astype(str) == tid].iloc[0]
        aux = aux_by_id.get(tid)
        if aux is None:
            rows.append({})
            continue
        scores = np.asarray(aux["deletion_score"], dtype=float)
        base_residue_prob = np.asarray(aux.get("base_residue_prob", []), dtype=float)
        n = len(scores)
        parsed = parse_chopping_star(str(bench_row["label"]))
        assign = residue_assignment(parsed.domains, n)
        non_domain = assign == 0
        boundary = boundary_mask_from_assign(assign, window=boundary_window)
        k_linker = int(non_domain.sum())
        bg = float(non_domain.mean()) if n else float("nan")
        if k_linker > 0 and np.isfinite(bg) and bg > 0:
            top = np.argsort(-scores)[:k_linker]
            precision_top = float(non_domain[top].mean())
            enrichment = precision_top / bg
        else:
            enrichment = float("nan")
        effective_budget = aux.get("deletion_budget", float("nan"))
        budget_fraction = aux.get("deletion_budget_fraction", float("nan"))
        pred_budget = (
            float(effective_budget)
            if np.isfinite(float(effective_budget))
            else (
                float(budget_fraction) * n
                if np.isfinite(float(budget_fraction))
                else (
                    float(np.sum(1.0 - base_residue_prob[:n]))
                    if len(base_residue_prob) >= n
                    else float("nan")
                )
            )
        )
        rows.append(
            {
                "non_domain_auroc": binary_auroc(non_domain, scores),
                "non_domain_auprc": binary_average_precision(non_domain, scores),
                "boundary_f1": topk_f1(boundary, scores),
                "linker_enrichment_topk": enrichment,
                "non_domain_fraction": bg,
                "true_deletion_budget": float(k_linker),
                "pred_deletion_budget": pred_budget,
                "budget_abs_error": abs(pred_budget - float(k_linker)) if np.isfinite(pred_budget) else float("nan"),
            }
        )
    extra = pd.DataFrame(rows)
    return pd.concat([metrics_df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def _metric_mean_ci(metrics_df: pd.DataFrame, col: str, mask: pd.Series) -> dict:
    vals = metrics_df.loc[mask, col].values.astype(float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    lo, hi = rb.bootstrap_ci(vals)
    return {"mean": float(np.mean(vals)), "ci_lo": lo, "ci_hi": hi, "n": int(len(vals))}


def deletion_summary(metrics_df: pd.DataFrame) -> dict:
    metric_cols = [
        "non_domain_auroc",
        "non_domain_auprc",
        "boundary_f1",
        "linker_enrichment_topk",
        "budget_abs_error",
    ]
    out = {}
    groups = {
        "overall": metrics_df["n_domains"] >= 1,
        "single": metrics_df["n_domains"] == 1,
        "multi": metrics_df["n_domains"] >= 2,
        "2+ domains": metrics_df["n_domains"] >= 2,
    }
    for name, mask in groups.items():
        out[name] = {col: _metric_mean_ci(metrics_df, col, mask) for col in metric_cols if col in metrics_df}
        if {"true_deletion_budget", "pred_deletion_budget"}.issubset(metrics_df.columns):
            sub = metrics_df.loc[mask, ["true_deletion_budget", "pred_deletion_budget"]].astype(float)
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) >= 2 and sub["true_deletion_budget"].std() > 0 and sub["pred_deletion_budget"].std() > 0:
                corr = float(np.corrcoef(sub["true_deletion_budget"], sub["pred_deletion_budget"])[0, 1])
            else:
                corr = float("nan")
            out[name]["budget_pearson"] = {"mean": corr, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(len(sub))}
    return out


def make_deletion_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_labels = [
        ("non_domain_auroc", "Non-domain AUROC"),
        ("non_domain_auprc", "Non-domain AUPRC"),
        ("boundary_f1", "Boundary F1"),
        ("linker_enrichment_topk", "Top-k linker enrichment"),
    ]
    groups = [
        ("Overall", metrics_df["n_domains"] >= 1),
        ("Single", metrics_df["n_domains"] == 1),
        ("2+ domains", metrics_df["n_domains"] >= 2),
    ]
    fig, axes = plt.subplots(len(groups), len(metric_labels), figsize=(12, 7))
    for r, (glabel, mask) in enumerate(groups):
        for c, (col, label) in enumerate(metric_labels):
            ax = axes[r, c]
            stat = _metric_mean_ci(metrics_df, col, mask)
            val = stat["mean"]
            if np.isnan(val):
                val = 0.0
            ax.bar([0], [val], color="#4C78A8", width=0.55)
            if np.isfinite(stat["ci_lo"]) and np.isfinite(stat["ci_hi"]):
                ax.errorbar([0], [val], yerr=[[val - stat["ci_lo"]], [stat["ci_hi"] - val]],
                            fmt="none", color="black", capsize=3)
            ax.set_xticks([])
            ax.set_title(f"{label}\n{glabel} (n={stat['n']})", fontsize=9)
            ax.text(0, val + 0.02 * max(1.0, val), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            if col != "linker_enrichment_topk":
                ax.set_ylim(0, 1.05)
    fig.suptitle("TED deletion-head residue metrics on CATH-1363", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved deletion figure -> {output_path}")


def run_diagnostic_shrink(bench_df, model, out_dir: Path, *, max_samples: int, deletion_fraction: float) -> None:
    if max_samples <= 0:
        return
    traces = []
    for _, row in bench_df.head(max_samples).iterrows():
        tid = str(row["target_id"])
        seq = str(row["sequence"])
        result = model.diagnostic_shrink(seq, deletion_fraction=deletion_fraction)
        parsed = parse_chopping_star(str(row["label"]))
        assign = residue_assignment(parsed.domains, len(seq))
        boundary = boundary_mask_from_assign(assign)
        for tr in result["trace"]:
            idx = int(tr["original_index"])
            traces.append(
                {
                    "target_id": tid,
                    **tr,
                    "true_non_domain": int(assign[idx] == 0) if idx < len(assign) else 0,
                    "near_boundary": int(boundary[idx]) if idx < len(boundary) else 0,
                }
            )
    if not traces:
        return
    trace_df = pd.DataFrame(traces)
    trace_path = out_dir / "diagnostic_shrink_traces.csv"
    trace_df.to_csv(trace_path, index=False)
    summary = {
        "n_targets": int(trace_df["target_id"].nunique()),
        "n_deleted": int(len(trace_df)),
        "deleted_true_non_domain_fraction": float(trace_df["true_non_domain"].mean()),
        "deleted_near_boundary_fraction": float(trace_df["near_boundary"].mean()),
    }
    (out_dir / "diagnostic_shrink_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    for tid, grp in trace_df.groupby("target_id"):
        ax.plot(grp["step"], grp["original_index"], marker="o", linewidth=1, label=tid)
    ax.set_xlabel("Deletion step")
    ax.set_ylabel("Original residue index")
    ax.set_title("Diagnostic shrink deletion order")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "diagnostic_shrink_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic shrink traces -> {trace_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark two-stage TEDPred on Chainsaw CATH-1363.")
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "two_stage_checkpoint.pt")
    p.add_argument("--benchmark-csv", type=Path,
                   default=THIS_DIR / "data" / "processed_with_cath" / "chainsaw_cath1363_with_cath_labels.csv")
    p.add_argument("--output-dir", type=Path, default=THIS_DIR / "results_two_stage")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-domain-len", type=int, default=20)
    p.add_argument("--cluster-method", type=str, default="connected_components",
                   choices=["connected_components", "spectral"])
    p.add_argument("--cath-decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--cath-beam-width", type=int, default=4)
    p.add_argument("--deletion-fusion-alpha", type=float, default=0.0)
    p.add_argument("--fusion-ablation", action="store_true",
                   help="Run auxiliary-only alpha=0 and fused alpha=--deletion-fusion-alpha.")
    p.add_argument("--boundary-window", type=int, default=2)
    p.add_argument("--diagnostic-shrink-samples", type=int, default=0)
    p.add_argument("--diagnostic-shrink-fraction", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=1700,
                   help="Truncate chains longer than this at inference (memory cap for the LxL head).")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--allow-legacy-benchmark", action="store_true",
                   help="Audit-only escape hatch for CSVs created with the former truth target.")
    p.add_argument("--ablation", action="store_true",
                   help="Compare CATH decode modes (greedy vs beam) sharing one segmentation pass.")
    return p.parse_args()


def _domain_count_table(metrics_df: pd.DataFrame, title: str = "") -> None:
    """Prominent per-domain-count breakdown (1, 2, 3, 4+, plus 2+ and overall)."""
    def m(sub, c):
        return float(sub[c].mean()) if c in sub.columns and len(sub) else float("nan")
    buckets = [
        ("overall", metrics_df["n_domains"] >= 1),
        ("1 dom",   metrics_df["n_domains"] == 1),
        ("2 dom",   metrics_df["n_domains"] == 2),
        ("3 dom",   metrics_df["n_domains"] == 3),
        ("4+ dom",  metrics_df["n_domains"] >= 4),
        ("2+ dom",  metrics_df["n_domains"] >= 2),
    ]
    if title:
        print(f"\n{title}")
    print("=" * 78)
    print(f"{'group':<10}{'n':>6}{'IoU':>8}{'Corr%':>8}{'BDS':>8}{'NDO':>8}{'CAT':>8}{'CAT-L':>8}{'parse':>8}")
    for name, mask in buckets:
        sub = metrics_df[mask]
        print(f"{name:<10}{len(sub):>6}{m(sub,'iou_chain'):>8.3f}{m(sub,'correct_prop'):>8.3f}"
              f"{m(sub,'boundary_distance_score'):>8.3f}{m(sub,'ndo'):>8.3f}"
              f"{m(sub,'correct_cath'):>8.3f}{m(sub,'cath_level_score'):>8.3f}"
              f"{m(sub,'pred_parse_ok'):>8.3f}")
    print("=" * 78)


def _chainsaw_comparison(bench_df: pd.DataFrame) -> dict:
    cs = {}
    if {"chainsaw_iou", "chainsaw_correct_prop", "chainsaw_boundary_distance_score"}.issubset(bench_df.columns):
        for gname, fn in [("1+ domains", lambda d: d["n_domains"] >= 1),
                          ("2+ domains", lambda d: d["n_domains"] >= 2)]:
            sub = bench_df[fn(bench_df)]
            cs[gname] = {
                "n": len(sub),
                "iou": float(sub["chainsaw_iou"].mean()),
                "correct_prop": float(sub["chainsaw_correct_prop"].mean()),
                "boundary_distance_score": float(sub["chainsaw_boundary_distance_score"].mean()),
            }
    return cs


def summarize_and_save(
    bench_df,
    pred_df,
    out_dir: Path,
    suffix: str = "",
    aux_df: pd.DataFrame | None = None,
    boundary_window: int = 2,
) -> pd.DataFrame:
    """Evaluate predictions, write per-chain metrics/summary/figures, return metrics."""
    metrics_df = rb.evaluate_predictions(bench_df, pred_df)
    metrics_df = add_deletion_metrics(
        bench_df, metrics_df, aux_df, boundary_window=boundary_window
    )
    metrics_df.to_csv(out_dir / f"per_chain_metrics{suffix}.csv", index=False)

    summary = rb.build_summary(metrics_df)
    summary["chainsaw_comparison"] = _chainsaw_comparison(bench_df)
    if aux_df is not None:
        summary["deletion_metrics"] = deletion_summary(metrics_df)
    with open(out_dir / f"summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    chainsaw_cols = [c for c in
                     ["target_id", "chainsaw_iou", "chainsaw_correct_prop",
                      "chainsaw_boundary_distance_score", "n_domains"]
                     if c in bench_df.columns]
    mws = metrics_df.merge(bench_df[chainsaw_cols], on="target_id", how="left", suffixes=("", "_bench"))
    if "n_domains_bench" in mws.columns:
        mws["n_domains"] = mws["n_domains_bench"]
        mws.drop(columns=["n_domains_bench"], inplace=True)
    if {"chainsaw_iou", "chainsaw_correct_prop", "chainsaw_boundary_distance_score"}.issubset(mws.columns):
        rb.make_figure(mws, bench_df, out_dir / f"figure6a{suffix}.png")
    rb.make_cath_figure(mws, out_dir / f"figure_cath{suffix}.png")
    if aux_df is not None:
        make_deletion_figure(metrics_df, out_dir / f"figure_deletion{suffix}.png")
    return metrics_df


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading benchmark CSV: {args.benchmark_csv}")
    bench_df = pd.read_csv(args.benchmark_csv)
    rb.validate_benchmark_contract(
        bench_df, allow_legacy=args.allow_legacy_benchmark
    )
    print(f"  {len(bench_df)} chains")

    # ---- ABLATION: auxiliary-only vs deletion-fused segmentation ---- #
    if args.fusion_ablation:
        print(f"\nLoading two-stage checkpoint: {args.checkpoint}")
        model, ckpt_args = load_two_stage(args.checkpoint, device)
        configs = [("aux_only", 0.0), ("fused", args.deletion_fusion_alpha)]
        for label, alpha in configs:
            print(f"\nFusion ablation: {label} deletion_fusion_alpha={alpha}")
            pred_df, aux_df = run_inference(
                bench_df,
                model,
                batch_size=args.batch_size,
                min_domain_len=args.min_domain_len,
                cluster_method=args.cluster_method,
                cath_decode=args.cath_decode,
                cath_beam_width=args.cath_beam_width,
                deletion_fusion_alpha=alpha,
                max_len=args.max_len,
                cache_path=out_dir / f"predictions_{label}.csv",
                cache_metadata=_cache_metadata(
                    bench_df,
                    args.checkpoint,
                    batch_size=args.batch_size,
                    min_domain_len=args.min_domain_len,
                    cluster_method=args.cluster_method,
                    cath_decode=args.cath_decode,
                    cath_beam_width=args.cath_beam_width,
                    deletion_fusion_alpha=alpha,
                    max_len=args.max_len,
                ),
            )
            metrics_df = summarize_and_save(
                bench_df,
                pred_df,
                out_dir,
                suffix=f"_{label}",
                aux_df=aux_df,
                boundary_window=args.boundary_window,
            )
            _domain_count_table(metrics_df, title=f"Fusion = {label}")
        run_diagnostic_shrink(
            bench_df,
            model,
            out_dir,
            max_samples=args.diagnostic_shrink_samples,
            deletion_fraction=args.diagnostic_shrink_fraction,
        )
        print("\nDone (fusion ablation).")
        return

    # ---- ABLATION: greedy vs beam CATH decode, shared segmentation ---- #
    if args.ablation:
        print(f"\nLoading two-stage checkpoint: {args.checkpoint}")
        model, ckpt_args = load_two_stage(args.checkpoint, device)
        modes = [("greedy", "greedy", 1), ("beam", "beam", args.cath_beam_width)]
        preds_by_mode = run_inference_multi(
            bench_df, model, modes, batch_size=args.batch_size,
            min_domain_len=args.min_domain_len, cluster_method=args.cluster_method,
            max_len=args.max_len,
        )
        for label, pred_df in preds_by_mode.items():
            pred_df.to_csv(out_dir / f"predictions_{label}.csv", index=False)
            metrics_df = summarize_and_save(bench_df, pred_df, out_dir, suffix=f"_{label}")
            _domain_count_table(metrics_df, title=f"CATH decode = {label}")
        print("\nNote: segmentation (IoU/BDS/NDO) is identical across CATH modes; "
              "only correct_cath / cath_level_score change. Beam is the recommended default.")
        print("\nDone (ablation).")
        return

    # ---- single configuration ---- #
    predictions_path = out_dir / "predictions.csv"
    if args.skip_inference and predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
        aux_path = predictions_path.with_name(predictions_path.stem + "_aux.parquet")
        aux_df = pd.read_parquet(aux_path) if aux_path.exists() else None
        model = None
    else:
        print(f"\nLoading two-stage checkpoint: {args.checkpoint}")
        model, ckpt_args = load_two_stage(args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Trainable head params: {n_params:.1f}M  esm={ckpt_args.get('esm_model_name')}  "
              f"cath_decode={args.cath_decode}")
        pred_df, aux_df = run_inference(
            bench_df, model,
            batch_size=args.batch_size, min_domain_len=args.min_domain_len,
            cluster_method=args.cluster_method, cath_decode=args.cath_decode,
            cath_beam_width=args.cath_beam_width,
            deletion_fusion_alpha=args.deletion_fusion_alpha,
            max_len=args.max_len,
            cache_path=predictions_path,
            cache_metadata=_cache_metadata(
                bench_df,
                args.checkpoint,
                batch_size=args.batch_size,
                min_domain_len=args.min_domain_len,
                cluster_method=args.cluster_method,
                cath_decode=args.cath_decode,
                cath_beam_width=args.cath_beam_width,
                deletion_fusion_alpha=args.deletion_fusion_alpha,
                max_len=args.max_len,
            ),
        )

    print("\nEvaluating predictions...")
    metrics_df = summarize_and_save(
        bench_df,
        pred_df,
        out_dir,
        aux_df=aux_df,
        boundary_window=args.boundary_window,
    )
    _domain_count_table(metrics_df, title=f"TED-Pred two-stage (CATH decode={args.cath_decode})")

    if args.diagnostic_shrink_samples > 0:
        if model is None:
            print("\nReloading model for diagnostic shrink.")
            model, _ = load_two_stage(args.checkpoint, device)
        run_diagnostic_shrink(
            bench_df,
            model,
            out_dir,
            max_samples=args.diagnostic_shrink_samples,
            deletion_fraction=args.diagnostic_shrink_fraction,
        )

    cs = _chainsaw_comparison(bench_df)
    if cs:
        print("\nChainsaw reference:")
        for g, d in cs.items():
            print(f"  {g:<12} n={d['n']:>5}  IoU={d['iou']:.3f}  Corr={d['correct_prop']:.3f}  BDS={d['boundary_distance_score']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
