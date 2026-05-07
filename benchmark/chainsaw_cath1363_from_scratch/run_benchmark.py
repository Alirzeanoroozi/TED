"""
Benchmark TED-Pred on the Chainsaw CATH-1363 test set.

Usage (from repo root with the 'ted' conda env):
    python benchmark/chainsaw_cath1363_from_scratch/run_benchmark.py \
        --checkpoint artifacts/transformer_checkpoint.pt \
        --data-parquet-dir data/all_parquet \
        --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed/chainsaw_cath1363_tedpred.csv \
        --output-dir benchmark/chainsaw_cath1363_from_scratch/results

Produces:
    results/predictions.csv          – per-chain predicted chopping_star
    results/per_chain_metrics.csv    – per-chain evaluation metrics
    results/summary.json             – aggregate mean metrics per domain group
    results/figure6a.png             – Chainsaw Fig-6a style comparison
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root or from the benchmark subdirectory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
for p in (str(SRC_DIR), str(BENCHMARK_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer_ import TextTokenizer
from model import TextToTextTransformer
from evaluate import grammar_guided_decode, greedy_decode
from data.dataset_ import _load_paths
from ted_eval import EvalConfig, evaluate_target


# ---------------------------------------------------------------------------
# Tokenizer reconstruction
# ---------------------------------------------------------------------------

def build_tokenizers(parquet_dir: Path) -> tuple[TextTokenizer, TextTokenizer]:
    """Rebuild tokenizers by fitting on the first 3 parquet shards.

    This exactly mirrors TEDDataModule.setup() in train_lightning.py.
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")
    fit_files = files[:3]
    print(f"Fitting tokenizers on {len(fit_files)} parquet shards: "
          f"{[f.name for f in fit_files]}")
    df = _load_paths([str(f) for f in fit_files])
    src_tok = TextTokenizer().fit(df["sequence"].astype(str).tolist())
    tgt_tok = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())
    print(f"  src vocab size: {src_tok.vocab_size}  "
          f"tgt vocab size: {tgt_tok.vocab_size}")
    return src_tok, tgt_tok


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TextToTextTransformer, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    a = ckpt.get("args", {})
    model = TextToTextTransformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=a.get("d_model", 256),
        nhead=a.get("nhead", 8),
        num_encoder_layers=a.get("num_encoder_layers", 4),
        num_decoder_layers=a.get("num_decoder_layers", 4),
        dim_feedforward=a.get("dim_feedforward", 1024),
        dropout=0.0,  # eval mode
        max_src_len=a.get("max_src_len", 1024),
        max_tgt_len=a.get("max_tgt_len", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, a


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_sequence(
    model: TextToTextTransformer,
    sequence: str,
    src_tok: TextTokenizer,
    tgt_tok: TextTokenizer,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
    grammar_guided: bool = True,
) -> str:
    seq = sequence[:max_src_len]
    src_ids = src_tok.encode(seq, add_sos_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    if grammar_guided:
        out_ids = grammar_guided_decode(
            model, src_tensor,
            src_pad_id=src_tok.pad_id,
            tgt_pad_id=tgt_tok.pad_id,
            sos_id=tgt_tok.sos_id,
            eos_id=tgt_tok.eos_id,
            max_len=max_tgt_len,
            device=device,
            token2id=tgt_tok.token2id,
            id2token=tgt_tok.id2token,
        )[0].tolist()
    else:
        out_ids = greedy_decode(
            model, src_tensor,
            src_pad_id=src_tok.pad_id,
            tgt_pad_id=tgt_tok.pad_id,
            sos_id=tgt_tok.sos_id,
            eos_id=tgt_tok.eos_id,
            max_len=max_tgt_len,
            device=device,
        )[0].tolist()

    return tgt_tok.decode(out_ids, strip_special=True)


def run_inference(
    df: pd.DataFrame,
    model: TextToTextTransformer,
    src_tok: TextTokenizer,
    tgt_tok: TextTokenizer,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
    grammar_guided: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Run inference on all rows; optionally resume from a cached predictions file."""
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        done_ids = set(cached["target_id"].astype(str))
        if done_ids.issuperset(set(df["target_id"].astype(str))):
            print(f"All {len(df)} predictions already cached at {cache_path}; skipping inference.")
            return cached
        print(f"Resuming from cache: {len(done_ids)} / {len(df)} done.")
    else:
        cached = None
        done_ids = set()

    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        tid = str(row["target_id"])
        if tid in done_ids:
            continue
        seq = str(row["sequence"])
        pred = predict_sequence(
            model, seq, src_tok, tgt_tok,
            max_src_len, max_tgt_len, device, grammar_guided,
        )
        rows.append({"target_id": tid, "pred_chopping_star": pred})

        if (len(rows) % 50 == 0) or (len(rows) + len(done_ids) == len(df)):
            elapsed = time.time() - t0
            total_done = len(rows) + len(done_ids)
            rate = len(rows) / elapsed if elapsed > 0 else float("inf")
            print(
                f"  [{total_done}/{len(df)}] {rate:.1f} seq/s  "
                f"last: {tid!r} -> {pred!r}"
            )

    new_df = pd.DataFrame(rows)
    if cached is not None and len(cached):
        result = pd.concat([cached[["target_id", "pred_chopping_star"]], new_df], ignore_index=True)
    else:
        result = new_df

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print(f"Saved predictions -> {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(
    bench_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge predictions with benchmark labels and compute per-chain metrics."""
    merged = bench_df.merge(pred_df, on="target_id", how="left")
    cfg = EvalConfig(iou_threshold=0.8, input_indexing="one_based_inclusive")

    records = []
    for _, row in merged.iterrows():
        label = row["label"]
        pred = row.get("pred_chopping_star", None)
        nres = int(row["nres"]) if not pd.isna(row["nres"]) else None
        seq = str(row["sequence"])

        if pd.isna(pred) or pred is None or str(pred).strip() == "":
            pred = "1-%d | -" % nres if nres else None

        m = evaluate_target(label, pred, nres=nres, sequence=seq, config=cfg)
        m["target_id"] = str(row["target_id"])
        m["domain_group"] = str(row["domain_group"])
        m["n_domains"] = int(row["n_domains"])
        m["pred_chopping_star"] = str(pred)
        m["label"] = str(label)
        records.append(m)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) confidence interval of the mean via bootstrap."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (math.nan, math.nan)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def group_stats(metrics_df: pd.DataFrame, col: str, group_mask: pd.Series) -> dict:
    vals = metrics_df.loc[group_mask, col].values.astype(float)
    mean = float(np.nanmean(vals))
    lo, hi = bootstrap_ci(vals)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int((~np.isnan(vals)).sum())}


# ---------------------------------------------------------------------------
# Figure generation (Chainsaw Fig 6a style)
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Chainsaw": "#E45E3F",   # red-orange (Chainsaw paper colour)
    "TED-Pred": "#5B8DB8",   # muted blue
}


def _bar_chart(
    ax,
    methods: list[str],
    values: list[float],
    ci_los: list[float],
    ci_his: list[float],
    ns: list[int],
    ylabel: str,
    title: str,
):
    x = np.arange(len(methods))
    width = 0.55
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    bars = ax.bar(x, values, width, color=colors, edgecolor="white", linewidth=0.8)

    # error bars
    yerr_lo = [v - lo for v, lo in zip(values, ci_los)]
    yerr_hi = [hi - v for v, hi in zip(values, ci_his)]
    ax.errorbar(
        x, values,
        yerr=[yerr_lo, yerr_hi],
        fmt="none", color="black", capsize=3, linewidth=1,
    )

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(
    tedpred_metrics: pd.DataFrame,
    chainsaw_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """3 metrics × 4 domain groups (Overall / Single / 2+), Chainsaw vs TED-Pred."""

    np.random.seed(42)

    # ---- align chainsaw df to tedpred test set ----
    # bench_df uses target_id; chainsaw raw CSV uses chain_id — handle both
    id_col = "chain_id" if "chain_id" in chainsaw_df.columns else "target_id"
    saw = chainsaw_df.set_index(id_col)

    metrics_cols = [
        ("iou_chain",               "chainsaw_iou",                     "IoU score"),
        ("correct_prop",            "chainsaw_correct_prop",             "Proportion correct domains"),
        ("boundary_distance_score", "chainsaw_boundary_distance_score",  "Domain boundary dist. score"),
    ]

    groups = [
        ("Overall",      tedpred_metrics["n_domains"] >= 1),
        ("Single domain", tedpred_metrics["n_domains"] == 1),
        ("2+ domains",   tedpred_metrics["n_domains"] >= 2),
    ]

    methods = ["Chainsaw", "TED-Pred"]

    n_rows = len(groups)
    n_cols = len(metrics_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9.5))

    for row_idx, (group_label, group_mask) in enumerate(groups):
        subset_ted = tedpred_metrics[group_mask]
        n = int(group_mask.sum())

        for col_idx, (ted_col, saw_col, metric_label) in enumerate(metrics_cols):
            ax = axes[row_idx, col_idx]

            # TED-Pred
            ted_vals = subset_ted[ted_col].values.astype(float)
            ted_mean = float(np.nanmean(ted_vals))
            ted_lo, ted_hi = bootstrap_ci(ted_vals)

            # Chainsaw (pre-computed per-chain scores carried in the merged df)
            saw_vals = subset_ted[saw_col].values.astype(float) if saw_col in subset_ted.columns else np.array([])
            if len(saw_vals) == 0 or np.all(np.isnan(saw_vals)):
                saw_mean, saw_lo, saw_hi = math.nan, math.nan, math.nan
            else:
                saw_mean = float(np.nanmean(saw_vals))
                saw_lo, saw_hi = bootstrap_ci(saw_vals)

            title_str = f"{group_label} (n={n})"
            _bar_chart(
                ax,
                methods,
                [saw_mean, ted_mean],
                [saw_lo, ted_lo],
                [saw_hi, ted_hi],
                [n, n],
                ylabel=metric_label if col_idx == 0 else "",
                title="",
            )

            # First row: show metric name above group label; other rows: group label only
            if row_idx == 0:
                ax.set_title(f"{metric_label}\n{title_str}", fontsize=8.5, pad=3)
            else:
                ax.set_title(title_str, fontsize=8.5, pad=3)

    # legend
    legend_patches = [
        mpatches.Patch(color=METHOD_COLORS["Chainsaw"], label="Chainsaw"),
        mpatches.Patch(color=METHOD_COLORS["TED-Pred"], label="TED-Pred (ours)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Assessing TED-Pred vs Chainsaw on CATH-1363 test set\n"
        "(bar plots show 95% CI via bootstrap)",
        fontsize=10, y=1.01,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {output_path}")


def make_cath_figure(
    tedpred_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart for CATH exact-match and hierarchical level score (TED-Pred only).

    Only meaningful when the benchmark CSV contains real CATH superfamily codes
    in the ground-truth labels (i.e. when run with the CATH-enriched CSV produced
    by add_cath_labels.py).  When labels contain '-' placeholders the bars will
    be 0 and a warning is printed instead of saving the figure.
    """
    np.random.seed(42)

    # cath_level_score > 0 only when gt contains real CATH codes (not '-' placeholders).
    # correct_cath alone is an unreliable indicator: it counts None==None as a match
    # when both pred and gt happen to emit '-', even with the original un-enriched CSV.
    has_cath = (
        "cath_level_score" in tedpred_metrics.columns
        and float(tedpred_metrics["cath_level_score"].mean()) > 0
    )
    if not has_cath:
        print(
            "  Skipping CATH figure: cath_level_score is 0 — ground-truth labels "
            "contain no real CATH superfamily codes.  Re-run with the CATH-enriched CSV "
            "(data/processed_with_cath/) to generate this figure."
        )
        return

    cath_metrics = [
        ("correct_cath",     "CATH exact match\n(correct superfamily proportion)"),
        ("cath_level_score", "CATH level score\n(hierarchical C.A.T.H match, 0–1)"),
    ]
    groups = [
        ("Overall",       tedpred_metrics["n_domains"] >= 1),
        ("Single domain", tedpred_metrics["n_domains"] == 1),
        ("2+ domains",    tedpred_metrics["n_domains"] >= 2),
    ]

    n_rows = len(groups)
    n_cols = len(cath_metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 9))

    for row_idx, (group_label, group_mask) in enumerate(groups):
        subset = tedpred_metrics[group_mask]
        n = int(group_mask.sum())

        for col_idx, (col, metric_label) in enumerate(cath_metrics):
            ax = axes[row_idx, col_idx]
            vals = subset[col].values.astype(float)
            mean = float(np.nanmean(vals))
            lo, hi = bootstrap_ci(vals)

            _bar_chart(
                ax,
                ["TED-Pred"],
                [mean],
                [lo],
                [hi],
                [n],
                ylabel=metric_label if col_idx == 0 else "",
                title="",
            )

            if row_idx == 0:
                ax.set_title(f"{metric_label.split(chr(10))[0]}\n{group_label} (n={n})", fontsize=8.5, pad=3)
            else:
                ax.set_title(f"{group_label} (n={n})", fontsize=8.5, pad=3)

    legend_patches = [
        mpatches.Patch(color=METHOD_COLORS["TED-Pred"], label="TED-Pred (ours)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=1,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "TED-Pred CATH accuracy on CATH-1363 test set\n"
        "(bar plots show 95% CI via bootstrap)",
        fontsize=10, y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CATH figure -> {output_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(metrics_df: pd.DataFrame) -> dict:
    np.random.seed(42)
    metric_cols = [
        "iou_chain", "correct_prop", "boundary_distance_score", "ndo",
        "domain_count_match", "correct_cath", "cath_level_score",
    ]
    summary = {"n_total": len(metrics_df), "groups": {}}

    group_defs = {
        "overall":    metrics_df["n_domains"] >= 1,
        "single":     metrics_df["n_domains"] == 1,
        "multi":      metrics_df["n_domains"] >= 2,
        "1+ domains": metrics_df["n_domains"] >= 1,
        "2+ domains": metrics_df["n_domains"] >= 2,
    }

    for gname, gmask in group_defs.items():
        sub = metrics_df[gmask]
        g = {"n": int(gmask.sum())}
        for col in metric_cols:
            if col in sub.columns:
                vals = sub[col].values.astype(float)
                mean = float(np.nanmean(vals))
                lo, hi = bootstrap_ci(vals)
                g[col] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        summary["groups"][gname] = g

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TED-Pred on Chainsaw CATH-1363 test set.")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "artifacts" / "transformer_checkpoint.pt")
    p.add_argument("--data-parquet-dir", type=Path,
                   default=REPO_ROOT / "data" / "all_parquet")
    p.add_argument("--benchmark-csv", type=Path,
                   default=Path(__file__).parent / "data" / "processed" / "chainsaw_cath1363_tedpred.csv")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "results")
    p.add_argument("--no-grammar", action="store_true",
                   help="Use plain greedy decoding instead of grammar-guided decoding.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Inference batch size (currently 1 is recommended).")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: 'cpu' or 'cuda'. Auto-detected by default.")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip inference and load from existing predictions.csv in output dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ----- device -----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- output dir -----
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load benchmark data -----
    print(f"\nLoading benchmark CSV: {args.benchmark_csv}")
    bench_df = pd.read_csv(args.benchmark_csv)
    print(f"  {len(bench_df)} chains  "
          f"(single: {(bench_df['domain_group']=='single').sum()}, "
          f"multi: {(bench_df['domain_group']=='multi').sum()})")

    predictions_path = out_dir / "predictions.csv"

    if args.skip_inference and predictions_path.exists():
        print(f"\nSkipping inference; loading {predictions_path}")
        pred_df = pd.read_csv(predictions_path)
    else:
        # ----- tokenizers -----
        print(f"\nBuilding tokenizers from {args.data_parquet_dir}")
        src_tok, tgt_tok = build_tokenizers(args.data_parquet_dir)

        # ----- model -----
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model, ckpt_args = load_model(args.checkpoint, device)
        max_src_len = ckpt_args.get("max_src_len", 1024)
        max_tgt_len = ckpt_args.get("max_tgt_len", 256)
        grammar_guided = not args.no_grammar
        decode_mode = "grammar-guided" if grammar_guided else "greedy"
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model params: {n_params:.1f}M  decode: {decode_mode}  "
              f"max_src_len: {max_src_len}  max_tgt_len: {max_tgt_len}")

        # ----- inference -----
        print(f"\nRunning inference on {len(bench_df)} sequences...")
        pred_df = run_inference(
            bench_df, model, src_tok, tgt_tok,
            max_src_len, max_tgt_len, device,
            grammar_guided=grammar_guided,
            cache_path=predictions_path,
        )

    # ----- evaluation -----
    print("\nEvaluating predictions...")
    metrics_df = evaluate_predictions(bench_df, pred_df)
    metrics_path = out_dir / "per_chain_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved per-chain metrics -> {metrics_path}")

    # ----- summary -----
    summary = build_summary(metrics_df)

    # add Chainsaw comparison to summary
    chainsaw_summary = {}
    for gname, gmask_fn in [
        ("1+ domains", lambda df: df["n_domains"] >= 1),
        ("2+ domains", lambda df: df["n_domains"] >= 2),
    ]:
        sub = bench_df[gmask_fn(bench_df)]
        chainsaw_summary[gname] = {
            "n": len(sub),
            "iou": float(sub["chainsaw_iou"].mean()),
            "correct_prop": float(sub["chainsaw_correct_prop"].mean()),
            "boundary_distance_score": float(sub["chainsaw_boundary_distance_score"].mean()),
        }
    summary["chainsaw_comparison"] = chainsaw_summary

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")

    # ----- print table -----
    has_cath = (
        "cath_level_score" in metrics_df.columns
        and float(metrics_df["cath_level_score"].mean()) > 0
    )
    cath_note = "" if has_cath else "  (CATH cols not meaningful — ground truth has no real CATH labels; use CATH-enriched CSV)"

    print("\n" + "=" * 85)
    print(f"{'Group':<15} {'n':>5}  {'IoU':>6}  {'Correct%':>8}  {'DBD':>6}  {'NDO':>6}"
          f"  {'CorrectCAT':>10}  {'CATH-Lvl':>8}")
    print("-" * 85)

    group_defs = [
        ("1+ domains", metrics_df["n_domains"] >= 1),
        ("2+ domains", metrics_df["n_domains"] >= 2),
        ("single",     metrics_df["n_domains"] == 1),
        ("multi",      metrics_df["n_domains"] >= 2),
    ]
    for gname, gmask in group_defs:
        sub = metrics_df[gmask]
        iou    = float(sub["iou_chain"].mean()) if "iou_chain" in sub else math.nan
        corr   = float(sub["correct_prop"].mean()) if "correct_prop" in sub else math.nan
        dbd    = float(sub["boundary_distance_score"].mean()) if "boundary_distance_score" in sub else math.nan
        ndo    = float(sub["ndo"].mean()) if "ndo" in sub else math.nan
        ccath  = float(sub["correct_cath"].mean()) if "correct_cath" in sub.columns else math.nan
        clvl   = float(sub["cath_level_score"].mean()) if "cath_level_score" in sub.columns else math.nan
        print(f"  TED-Pred {gname:<12} {len(sub):>5}  {iou:>6.3f}  {corr:>8.3f}  "
              f"{dbd:>6.3f}  {ndo:>6.3f}  {ccath:>10.3f}  {clvl:>8.3f}")

    print()
    for gname, data in chainsaw_summary.items():
        print(f"  Chainsaw {gname:<12} {data['n']:>5}  {data['iou']:>6.3f}  "
              f"{data['correct_prop']:>8.3f}  {data['boundary_distance_score']:>6.3f}"
              f"  {'N/A':>10}  {'N/A':>8}")
    print("=" * 85)
    if cath_note:
        print(cath_note)

    # ----- figure -----
    print("\nGenerating figure...")
    # Merge chainsaw scores into metrics_df for figure generation
    chainsaw_cols = ["target_id", "chainsaw_iou", "chainsaw_correct_prop", "chainsaw_boundary_distance_score", "n_domains"]
    metrics_with_saw = metrics_df.merge(
        bench_df[chainsaw_cols],
        on="target_id",
        how="left",
        suffixes=("", "_bench"),
    )
    # n_domains from bench_df is authoritative
    if "n_domains_bench" in metrics_with_saw.columns:
        metrics_with_saw["n_domains"] = metrics_with_saw["n_domains_bench"]
        metrics_with_saw.drop(columns=["n_domains_bench"], inplace=True)

    fig_path = out_dir / "figure6a.png"
    make_figure(metrics_with_saw, bench_df, fig_path)

    cath_fig_path = out_dir / "figure_cath.png"
    make_cath_figure(metrics_with_saw, cath_fig_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
