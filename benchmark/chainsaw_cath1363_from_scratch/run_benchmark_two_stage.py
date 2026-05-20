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
import json
import sys
import time
from pathlib import Path

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
from two_stage.infer import load_two_stage, predict_chopping_star_batch  # noqa: E402


def run_inference(
    bench_df: pd.DataFrame,
    model,
    *,
    batch_size: int,
    min_domain_len: int,
    cluster_method: str,
    max_len: int,
    cache_path: Path | None,
) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(cached["target_id"].astype(str)).issuperset(set(bench_df["target_id"].astype(str))):
            print(f"All predictions cached at {cache_path}; skipping inference.")
            return cached

    target_ids = bench_df["target_id"].astype(str).tolist()
    sequences = bench_df["sequence"].astype(str).tolist()

    t0 = time.time()
    preds = predict_chopping_star_batch(
        model, sequences, batch_size=batch_size,
        cluster_method=cluster_method, min_domain_len=min_domain_len, max_len=max_len,
    )
    dt = time.time() - t0
    print(f"Inference on {len(sequences)} chains in {dt:.1f}s ({len(sequences)/max(dt,1e-9):.1f} seq/s)")

    pred_df = pd.DataFrame({"target_id": target_ids, "pred_chopping_star": preds})
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(cache_path, index=False)
        print(f"Saved predictions -> {cache_path}")
    return pred_df


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
    p.add_argument("--max-len", type=int, default=1700,
                   help="Truncate chains longer than this at inference (memory cap for the LxL head).")
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.csv"

    print(f"\nLoading benchmark CSV: {args.benchmark_csv}")
    bench_df = pd.read_csv(args.benchmark_csv)
    print(f"  {len(bench_df)} chains")

    if args.skip_inference and predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
    else:
        print(f"\nLoading two-stage checkpoint: {args.checkpoint}")
        model, ckpt_args = load_two_stage(args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Trainable head params: {n_params:.1f}M  esm={ckpt_args.get('esm_model_name')}")
        pred_df = run_inference(
            bench_df, model,
            batch_size=args.batch_size, min_domain_len=args.min_domain_len,
            cluster_method=args.cluster_method, max_len=args.max_len,
            cache_path=predictions_path,
        )

    print("\nEvaluating predictions...")
    metrics_df = rb.evaluate_predictions(bench_df, pred_df)
    metrics_df.to_csv(out_dir / "per_chain_metrics.csv", index=False)

    summary = rb.build_summary(metrics_df)
    chainsaw_summary = {}
    for gname, gmask_fn in [
        ("1+ domains", lambda df: df["n_domains"] >= 1),
        ("2+ domains", lambda df: df["n_domains"] >= 2),
    ]:
        sub = bench_df[gmask_fn(bench_df)]
        if {"chainsaw_iou", "chainsaw_correct_prop", "chainsaw_boundary_distance_score"}.issubset(bench_df.columns):
            chainsaw_summary[gname] = {
                "n": len(sub),
                "iou": float(sub["chainsaw_iou"].mean()),
                "correct_prop": float(sub["chainsaw_correct_prop"].mean()),
                "boundary_distance_score": float(sub["chainsaw_boundary_distance_score"].mean()),
            }
    summary["chainsaw_comparison"] = chainsaw_summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {out_dir / 'summary.json'}")

    # Figures (reuse baseline plotting; needs chainsaw cols merged in).
    chainsaw_cols = [c for c in
                     ["target_id", "chainsaw_iou", "chainsaw_correct_prop",
                      "chainsaw_boundary_distance_score", "n_domains"]
                     if c in bench_df.columns]
    metrics_with_saw = metrics_df.merge(bench_df[chainsaw_cols], on="target_id", how="left", suffixes=("", "_bench"))
    if "n_domains_bench" in metrics_with_saw.columns:
        metrics_with_saw["n_domains"] = metrics_with_saw["n_domains_bench"]
        metrics_with_saw.drop(columns=["n_domains_bench"], inplace=True)

    if {"chainsaw_iou", "chainsaw_correct_prop", "chainsaw_boundary_distance_score"}.issubset(metrics_with_saw.columns):
        rb.make_figure(metrics_with_saw, bench_df, out_dir / "figure6a.png")
    rb.make_cath_figure(metrics_with_saw, out_dir / "figure_cath.png")

    # Console table
    print("\n" + "=" * 70)
    print(f"{'Group':<14}{'n':>6}{'IoU':>8}{'Corr%':>8}{'BDS':>8}{'NDO':>8}{'CAT':>8}{'CAT-L':>8}")
    for gname, gmask in [
        ("1+ domains", metrics_df["n_domains"] >= 1),
        ("2+ domains", metrics_df["n_domains"] >= 2),
        ("single", metrics_df["n_domains"] == 1),
    ]:
        sub = metrics_df[gmask]
        def m(c):
            return float(sub[c].mean()) if c in sub.columns and len(sub) else float("nan")
        print(f"{gname:<14}{len(sub):>6}{m('iou_chain'):>8.3f}{m('correct_prop'):>8.3f}"
              f"{m('boundary_distance_score'):>8.3f}{m('ndo'):>8.3f}{m('correct_cath'):>8.3f}{m('cath_level_score'):>8.3f}")
    print("=" * 70)
    print("\nDone.")


if __name__ == "__main__":
    main()
