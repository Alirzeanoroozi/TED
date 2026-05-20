#!/usr/bin/env python3
"""Audit CATH granularity / indexing consistency between the TED training labels
and the API-backfilled CHAINSAW-CATH1363 benchmark labels.

A silent mismatch (e.g. training labels are mostly 3-level while the benchmark is
4-level, or different segment index bases, or benchmark superfamilies absent from
the training vocabulary) would cap CATH exact-match no matter how good the model
is. This script surfaces such issues.

Usage:
    python benchmark/chainsaw_cath1363_from_scratch/audit_cath_consistency.py \
        --train-parquet-dir data/all_parquet \
        --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv \
        --train-sample 50000 --out audit_report.json

    python .../audit_cath_consistency.py --self-test     # logic checks, no data
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

_SEG_RE = re.compile(r"^(\d+)-(\d+)$")
_FULL_CATH_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


# --------------------------------------------------------------------------- #
# Parsing helpers (pure; unit-testable)
# --------------------------------------------------------------------------- #
def domain_codes_from_chopping(chopping: str) -> List[Optional[str]]:
    """One entry per domain: its CATH code, or None if unlabeled ('-')."""
    out: List[Optional[str]] = []
    for chunk in str(chopping).split("*"):
        if "|" not in chunk:
            continue
        code = chunk.split("|", 1)[1].strip()
        out.append(None if code in ("", "-") else code)
    return out


def segment_starts_from_chopping(chopping: str) -> List[int]:
    starts: List[int] = []
    for chunk in str(chopping).split("*"):
        bounds = chunk.split("|", 1)[0]
        for seg in bounds.split("_"):
            m = _SEG_RE.match(seg.strip())
            if m:
                starts.append(int(m.group(1)))
    return starts


def analyze_codes(codes: List[Optional[str]]) -> Dict:
    labeled = [c for c in codes if c]
    level_hist = Counter(c.count(".") + 1 for c in labeled)
    n_full = sum(1 for c in labeled if _FULL_CATH_RE.match(c))
    classes = Counter(c.split(".")[0] for c in labeled)
    return {
        "n_domains_total": len(codes),
        "n_labeled": len(labeled),
        "n_unlabeled": len(codes) - len(labeled),
        "level_histogram": dict(sorted(level_hist.items())),
        "pct_4level_of_labeled": (n_full / len(labeled)) if labeled else 0.0,
        "n_unique_superfamily": len(set(labeled)),
        "class_distribution": dict(sorted(classes.items())),
    }


def index_base_guess(starts: List[int]) -> Optional[int]:
    if not starts:
        return None
    return 0 if min(starts) == 0 else 1


def vocab_by_level(codes: List[Optional[str]]) -> List[set]:
    """Per-level set of cumulative dotted prefixes seen in `codes`."""
    levels: List[set] = [set(), set(), set(), set()]
    for c in codes:
        if not c:
            continue
        parts = [p for p in c.split(".") if p]
        for i in range(min(len(parts), 4)):
            levels[i].add(".".join(parts[: i + 1]))
    return levels


def compare(train_codes: List[Optional[str]], bench_codes: List[Optional[str]]) -> Dict:
    """Coverage of benchmark CATH labels by the training-label vocabulary."""
    train_levels = vocab_by_level(train_codes)
    bench_labeled = [c for c in bench_codes if c]
    if not bench_labeled:
        return {"note": "no labeled benchmark domains"}
    level_names = ["C", "A", "T", "H"]
    coverage = {}
    for i, name in enumerate(level_names):
        bench_level = set()
        for c in bench_labeled:
            parts = [p for p in c.split(".") if p]
            if len(parts) > i:
                bench_level.add(".".join(parts[: i + 1]))
        if bench_level:
            seen = len(bench_level & train_levels[i])
            coverage[name] = {
                "benchmark_unique": len(bench_level),
                "in_training": seen,
                "coverage_frac": seen / len(bench_level),
            }
    # fraction of benchmark *domains* whose full H code is in training H-vocab
    full_in_train = sum(1 for c in bench_labeled if c in train_levels[3])
    coverage["domain_level_H_recall"] = full_in_train / len(bench_labeled)
    return coverage


def flags(train_stats, bench_stats, cmp_stats, train_base, bench_base) -> List[str]:
    out: List[str] = []
    if train_stats and bench_stats:
        tp, bp = train_stats["pct_4level_of_labeled"], bench_stats["pct_4level_of_labeled"]
        if abs(tp - bp) > 0.1:
            out.append(f"GRANULARITY mismatch: train {tp:.0%} 4-level vs benchmark {bp:.0%} 4-level")
    if train_base is not None and bench_base is not None and train_base != bench_base:
        out.append(f"INDEX-BASE mismatch: train base={train_base} vs benchmark base={bench_base}")
    if cmp_stats and "H" in cmp_stats and isinstance(cmp_stats["H"], dict):
        hc = cmp_stats["H"]["coverage_frac"]
        if hc < 0.5:
            out.append(f"LOW H-COVERAGE: only {hc:.0%} of benchmark superfamilies appear in training "
                       "labels -> exact-match is capped near this regardless of model quality")
    if not out:
        out.append("no major inconsistencies detected")
    return out


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _chopping_series(df):
    col = "chopping_star" if "chopping_star" in df.columns else ("label" if "label" in df.columns else None)
    if col is None:
        raise ValueError(f"No chopping_star/label column; have {list(df.columns)}")
    return df[col].astype(str)


def load_training(train_dir: Path, sample: int):
    import pandas as pd
    files = sorted(train_dir.glob("*.parquet"))
    if not files:
        return None, None
    codes: List[Optional[str]] = []
    starts: List[int] = []
    rows = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as exc:
            print(f"[warn] could not read {f.name}: {exc}")
            continue
        for cs in _chopping_series(df):
            codes.extend(domain_codes_from_chopping(cs))
            starts.extend(segment_starts_from_chopping(cs))
            rows += 1
            if sample and rows >= sample:
                return codes, starts
    return codes, starts


def load_benchmark(csv_path: Path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    series = _chopping_series(df)
    codes, starts = [], []
    for cs in series:
        codes.extend(domain_codes_from_chopping(cs))
        starts.extend(segment_starts_from_chopping(cs))
    return codes, starts


# --------------------------------------------------------------------------- #
def _self_test():
    train = ["3.40.50.300", "3.40.50.300", "1.10.10.10", None, "2.60.40.10"]
    bench = ["3.40.50.300", "1.10.10.10", "9.99.99.99"]  # last unseen in train
    ts, bs = analyze_codes(train), analyze_codes(bench)
    assert ts["n_unlabeled"] == 1 and ts["pct_4level_of_labeled"] == 1.0
    cmp = compare(train, bench)
    assert cmp["domain_level_H_recall"] == 2 / 3, cmp     # 9.99.99.99 unseen
    assert cmp["C"]["coverage_frac"] == 2 / 3, cmp        # classes {3,1,9}; 9 unseen
    assert index_base_guess([1, 5, 60]) == 1 and index_base_guess([0, 4]) == 0
    print("self-test OK:", json.dumps(cmp, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet-dir", type=Path, default=REPO_ROOT / "data" / "all_parquet")
    p.add_argument("--benchmark-csv", type=Path,
                   default=THIS_DIR / "data" / "processed_with_cath" / "chainsaw_cath1363_with_cath_labels.csv")
    p.add_argument("--train-sample", type=int, default=50000, help="Max training chains to scan (0=all).")
    p.add_argument("--out", type=Path, default=THIS_DIR / "cath_audit_report.json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    report: Dict = {}

    bench_codes, bench_starts = (None, None)
    if args.benchmark_csv.exists():
        bench_codes, bench_starts = load_benchmark(args.benchmark_csv)
        report["benchmark"] = analyze_codes(bench_codes)
        report["benchmark"]["index_base_guess"] = index_base_guess(bench_starts)
        print("BENCHMARK:", json.dumps(report["benchmark"], indent=2))
    else:
        print(f"[warn] benchmark CSV not found: {args.benchmark_csv}")

    train_codes, train_starts = (None, None)
    if args.train_parquet_dir.exists():
        train_codes, train_starts = load_training(args.train_parquet_dir, args.train_sample)
        if train_codes:
            report["training"] = analyze_codes(train_codes)
            report["training"]["index_base_guess"] = index_base_guess(train_starts)
            print("\nTRAINING:", json.dumps(report["training"], indent=2))
    else:
        print(f"[info] training parquet dir not found ({args.train_parquet_dir}); "
              "auditing benchmark side only (run on the cluster for the full comparison).")

    if train_codes and bench_codes:
        report["coverage"] = compare(train_codes, bench_codes)
        print("\nCOVERAGE (benchmark labels found in training vocab):",
              json.dumps(report["coverage"], indent=2))

    report["flags"] = flags(
        report.get("training"), report.get("benchmark"), report.get("coverage"),
        index_base_guess(train_starts) if train_starts else None,
        index_base_guess(bench_starts) if bench_starts else None,
    )
    print("\nFLAGS:")
    for f in report["flags"]:
        print(f"  - {f}")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
