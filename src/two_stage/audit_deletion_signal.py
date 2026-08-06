"""Audit how much TED supervision exists for a SCISOR-style deletion signal.

The proposed diffusion auxiliary objective needs residues that should be
deleted from a full protein to recover its domain-bearing core. In TED labels,
those are exactly residues not covered by any parsed domain interval.

This script scans TED parquet shards and reports:
  * how many residues are outside domains,
  * whether that signal survives the training max_len truncation,
  * how the signal varies by domain count, source shard, and length.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from two_stage.targets import ParsedDomain, parse_chopping_star


def resolve_parquet_files(path: Path) -> list[Path]:
    path = path.expanduser()
    path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a parquet file, got {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files under {path}")
        return files
    raise FileNotFoundError(f"Data path does not exist: {path}")


def annotation_column(columns: Iterable[str]) -> str:
    cols = set(columns)
    if "chopping_star" in cols:
        return "chopping_star"
    if "label" in cols:
        return "label"
    raise ValueError("Expected an annotation column named 'chopping_star' or 'label'.")


def source_kind(path: Path) -> str:
    return "augmented" if "augmented" in path.stem else "original"


def clipped_intervals(domains: list[ParsedDomain], nres: int) -> tuple[list[tuple[int, int]], int]:
    """Return merged covered intervals and count of domains visible in [0, nres)."""
    intervals: list[tuple[int, int]] = []
    present_domains = 0
    for dom in domains:
        dom_present = False
        for s, e in dom.segments:
            s_c, e_c = max(0, s), min(nres, e)
            if e_c > s_c:
                intervals.append((s_c, e_c))
                dom_present = True
        if dom_present:
            present_domains += 1

    if not intervals:
        return [], present_domains

    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged, present_domains


def non_domain_runs(covered: list[tuple[int, int]], nres: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    pos = 0
    for s, e in covered:
        if s > pos:
            runs.append((pos, s))
        pos = max(pos, e)
    if pos < nres:
        runs.append((pos, nres))
    return runs


def summarize_view(domains: list[ParsedDomain], nres: int) -> dict:
    covered, domain_count = clipped_intervals(domains, nres)
    domain_res = sum(e - s for s, e in covered)
    non_runs = non_domain_runs(covered, nres)
    non_res = nres - domain_res

    terminal_non = 0
    internal_non = 0
    if non_runs:
        for s, e in non_runs:
            if not covered or e <= covered[0][0] or s >= covered[-1][1]:
                terminal_non += e - s
            else:
                internal_non += e - s

    max_run = max((e - s for s, e in non_runs), default=0)
    return {
        "nres": nres,
        "domain_count": domain_count,
        "domain_residues": domain_res,
        "non_domain_residues": non_res,
        "non_domain_fraction": (non_res / nres) if nres else 0.0,
        "terminal_non_domain_residues": terminal_non,
        "internal_non_domain_residues": internal_non,
        "non_domain_run_count": len(non_runs),
        "max_non_domain_run": max_run,
        "has_non_domain": int(non_res > 0),
    }


def domain_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    return "5+"


def length_bucket(n: int) -> str:
    edges = [100, 200, 400, 800, 1022, 1500, 2500]
    lo = 1
    for hi in edges:
        if n <= hi:
            return f"{lo}-{hi}"
        lo = hi + 1
    return "2501+"


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: float("nan") for k in ["p00", "p05", "p25", "p50", "p75", "p95", "p100"]}
    qs = np.quantile(np.asarray(values, dtype=float), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "p00": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p100": float(qs[6]),
    }


def aggregate(rows: list[dict], label: str) -> dict:
    n = len(rows)
    total_res = sum(r["nres"] for r in rows)
    total_non = sum(r["non_domain_residues"] for r in rows)
    total_domain = sum(r["domain_residues"] for r in rows)
    total_terminal = sum(r["terminal_non_domain_residues"] for r in rows)
    total_internal = sum(r["internal_non_domain_residues"] for r in rows)
    fracs = [r["non_domain_fraction"] for r in rows]
    return {
        "group": label,
        "chains": n,
        "total_residues": total_res,
        "domain_residues": total_domain,
        "non_domain_residues": total_non,
        "weighted_non_domain_fraction": (total_non / total_res) if total_res else float("nan"),
        "mean_chain_non_domain_fraction": float(np.mean(fracs)) if fracs else float("nan"),
        "median_chain_non_domain_fraction": float(np.median(fracs)) if fracs else float("nan"),
        "chains_with_any_non_domain": sum(r["has_non_domain"] for r in rows),
        "chains_with_any_non_domain_fraction": (sum(r["has_non_domain"] for r in rows) / n) if n else float("nan"),
        "chains_all_domain": sum(1 for r in rows if r["nres"] > 0 and r["non_domain_residues"] == 0),
        "terminal_non_domain_residues": total_terminal,
        "internal_non_domain_residues": total_internal,
        "mean_non_domain_run_count": float(np.mean([r["non_domain_run_count"] for r in rows])) if rows else float("nan"),
        "mean_max_non_domain_run": float(np.mean([r["max_non_domain_run"] for r in rows])) if rows else float("nan"),
        **{f"non_domain_fraction_{k}": v for k, v in quantiles(fracs).items()},
    }


def write_group_csv(rows: list[dict], key: str, out_path: Path) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out = [aggregate(group_rows, group) for group, group_rows in sorted(grouped.items())]
    pd.DataFrame(out).to_csv(out_path, index=False)
    return out


def write_markdown(summary: dict, out_path: Path) -> None:
    full = summary["views"]["full"]["overall"]
    capped = summary["views"]["capped"]["overall"]
    verdict = summary["verdict"]
    lines = [
        "# TED Deletion Signal Audit",
        "",
        f"Data path: `{summary['data_path']}`",
        f"Max training length: `{summary['max_len']}`",
        f"Rows scanned: `{summary['rows_scanned']:,}`",
        f"Parse failures: `{summary['parse_failures']:,}`",
        "",
        "## Overall",
        "",
        "| View | Chains | Residues | Non-domain residues | Weighted non-domain fraction | Chains with any non-domain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Full | {full['chains']:,} | {full['total_residues']:,} | "
            f"{full['non_domain_residues']:,} | {full['weighted_non_domain_fraction']:.4f} | "
            f"{full['chains_with_any_non_domain_fraction']:.4f} |"
        ),
        (
            f"| Capped | {capped['chains']:,} | {capped['total_residues']:,} | "
            f"{capped['non_domain_residues']:,} | {capped['weighted_non_domain_fraction']:.4f} | "
            f"{capped['chains_with_any_non_domain_fraction']:.4f} |"
        ),
        "",
        "## Verdict",
        "",
        f"- {verdict}",
        "",
        "## Notes",
        "",
        "- `non_domain_residues` are residues not covered by any parsed TED domain interval.",
        "- `capped` reflects the current two-stage training view after sequence truncation.",
        "- See the CSV files in this directory for group-by-domain-count, source, and length splits.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parquet-folder", type=Path, default=PROJECT_ROOT / "data" / "all_parquet")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis" / "deletion_signal_audit")
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional smoke-test row limit.")
    args = parser.parse_args()

    files = resolve_parquet_files(args.data_parquet_folder)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows: list[dict] = []
    capped_rows: list[dict] = []
    parse_failures = 0
    rows_scanned = 0

    for file_idx, path in enumerate(files, start=1):
        df = pd.read_parquet(path)
        ann_col = annotation_column(df.columns)
        if "sequence" not in df.columns:
            raise ValueError(f"Missing sequence column in {path}")
        kind = source_kind(path)
        print(f"[{file_idx:03d}/{len(files):03d}] {path.name}: {len(df):,} rows ({kind})")

        for row in df[["sequence", ann_col]].itertuples(index=False):
            if args.limit_rows is not None and rows_scanned >= args.limit_rows:
                break
            seq = str(row[0]).strip()
            ann = str(row[1]).strip()
            n_full = len(seq)
            parsed = parse_chopping_star(ann)
            if not parsed.parse_ok:
                parse_failures += 1
            base = {
                "source_kind": kind,
                "source_file": path.name,
                "parse_ok": int(parsed.parse_ok),
            }

            full = summarize_view(parsed.domains, n_full)
            full.update(base)
            full["domain_bucket"] = domain_bucket(full["domain_count"])
            full["length_bucket"] = length_bucket(full["nres"])
            full_rows.append(full)

            n_cap = min(n_full, args.max_len)
            capped = summarize_view(parsed.domains, n_cap)
            capped.update(base)
            capped["domain_bucket"] = domain_bucket(capped["domain_count"])
            capped["length_bucket"] = length_bucket(capped["nres"])
            capped_rows.append(capped)

            rows_scanned += 1
        if args.limit_rows is not None and rows_scanned >= args.limit_rows:
            break

    view_rows = {"full": full_rows, "capped": capped_rows}
    summary = {
        "data_path": str(args.data_parquet_folder),
        "max_len": args.max_len,
        "rows_scanned": rows_scanned,
        "parse_failures": parse_failures,
        "views": {},
    }

    for view, rows in view_rows.items():
        view_dir = out_dir / view
        view_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_parquet(view_dir / "per_chain.parquet", index=False)
        overall = aggregate(rows, "all")
        by_source = write_group_csv(rows, "source_kind", view_dir / "by_source.csv")
        by_domain = write_group_csv(rows, "domain_bucket", view_dir / "by_domain_count.csv")
        by_length = write_group_csv(rows, "length_bucket", view_dir / "by_length.csv")

        source_domain_rows = []
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["source_kind"], row["domain_bucket"])].append(row)
        for (src, dom), group_rows in sorted(grouped.items()):
            agg = aggregate(group_rows, f"{src}|{dom}")
            agg["source_kind"] = src
            agg["domain_bucket"] = dom
            source_domain_rows.append(agg)
        pd.DataFrame(source_domain_rows).to_csv(view_dir / "by_source_and_domain_count.csv", index=False)

        summary["views"][view] = {
            "overall": overall,
            "by_source": by_source,
            "by_domain_count": by_domain,
            "by_length": by_length,
        }

    capped = summary["views"]["capped"]["overall"]
    chain_signal = capped["chains_with_any_non_domain_fraction"]
    residue_signal = capped["weighted_non_domain_fraction"]
    if residue_signal < 0.05 or chain_signal < 0.50:
        verdict = (
            "Real TED non-domain residues are sparse in the training view; "
            "oversample the empirical full-protein state and compare artificial "
            "insertion only as a separately labelled control."
        )
    elif residue_signal < 0.10:
        verdict = (
            "Real TED non-domain residues provide a usable but modest deletion signal; "
            "prioritize physically shortened empirical states and keep artificial "
            "insertion as an ablation."
        )
    else:
        verdict = (
            "Real TED labels contain enough non-domain residue signal to train a "
            "supervised empirical deletion path; artificial noising is optional and "
            "must be evaluated separately."
        )
    summary["verdict"] = verdict

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(summary, out_dir / "README.md")
    print(f"[done] wrote audit to {out_dir}")
    print(verdict)


if __name__ == "__main__":
    main()
