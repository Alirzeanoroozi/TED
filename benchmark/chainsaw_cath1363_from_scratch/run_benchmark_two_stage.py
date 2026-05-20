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
    cath_decode: str,
    cath_beam_width: int,
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
        cluster_method=cluster_method, min_domain_len=min_domain_len,
        cath_decode=cath_decode, cath_beam_width=cath_beam_width, max_len=max_len,
    )
    dt = time.time() - t0
    print(f"Inference on {len(sequences)} chains in {dt:.1f}s ({len(sequences)/max(dt,1e-9):.1f} seq/s)")

    pred_df = pd.DataFrame({"target_id": target_ids, "pred_chopping_star": preds})
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(cache_path, index=False)
        print(f"Saved predictions -> {cache_path}")
    return pred_df


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
    p.add_argument("--max-len", type=int, default=1700,
                   help="Truncate chains longer than this at inference (memory cap for the LxL head).")
    p.add_argument("--skip-inference", action="store_true")
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


def summarize_and_save(bench_df, pred_df, out_dir: Path, suffix: str = "") -> pd.DataFrame:
    """Evaluate predictions, write per-chain metrics/summary/figures, return metrics."""
    metrics_df = rb.evaluate_predictions(bench_df, pred_df)
    metrics_df.to_csv(out_dir / f"per_chain_metrics{suffix}.csv", index=False)

    summary = rb.build_summary(metrics_df)
    summary["chainsaw_comparison"] = _chainsaw_comparison(bench_df)
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
    return metrics_df


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading benchmark CSV: {args.benchmark_csv}")
    bench_df = pd.read_csv(args.benchmark_csv)
    print(f"  {len(bench_df)} chains")

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
    else:
        print(f"\nLoading two-stage checkpoint: {args.checkpoint}")
        model, ckpt_args = load_two_stage(args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Trainable head params: {n_params:.1f}M  esm={ckpt_args.get('esm_model_name')}  "
              f"cath_decode={args.cath_decode}")
        pred_df = run_inference(
            bench_df, model,
            batch_size=args.batch_size, min_domain_len=args.min_domain_len,
            cluster_method=args.cluster_method, cath_decode=args.cath_decode,
            cath_beam_width=args.cath_beam_width, max_len=args.max_len,
            cache_path=predictions_path,
        )

    print("\nEvaluating predictions...")
    metrics_df = summarize_and_save(bench_df, pred_df, out_dir)
    _domain_count_table(metrics_df, title=f"TED-Pred two-stage (CATH decode={args.cath_decode})")

    cs = _chainsaw_comparison(bench_df)
    if cs:
        print("\nChainsaw reference:")
        for g, d in cs.items():
            print(f"  {g:<12} n={d['n']:>5}  IoU={d['iou']:.3f}  Corr={d['correct_prop']:.3f}  BDS={d['boundary_distance_score']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
