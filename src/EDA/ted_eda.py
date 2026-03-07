#!/usr/bin/env python3
"""
TED EDA pipeline for chain/domain parsing and publication-ready summary plots.

Dependencies:
  - pandas
  - numpy
  - matplotlib

Run from project root:
  python3 scripts/ted_eda.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_VERSION = "1.0.0"
RNG_SEED = 42

SEGMENT_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
DOMAIN_SPLIT_RE = re.compile(r"\s*\*\s*")


@dataclass
class TreePruneConfig:
    """Controls CATH hierarchy pruning for the tree figure."""

    top_classes: int = 6
    top_arch_per_class: int = 4
    top_topo_per_arch: int = 3
    top_superfamily_per_topo: int = 2


@dataclass
class CircularPlotConfig:
    """Controls pruning and labeling for circular CATH overview plot."""

    max_leaves: int = 60
    leaf_label_top_n: int = 30
    leaf_level: str = "topology"
    min_label_angle_deg: float = 8.0
    label_target: int = 30
    label_max_candidates: int = 60


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TED EDA parsing + plotting pipeline.")
    parser.add_argument("--chains-csv", type=Path, default=Path("data/chains.ted.csv"))
    parser.add_argument(
        "--missing-a", type=Path, default=Path("data/missing_accessions_TEDseqA.txt")
    )
    parser.add_argument(
        "--missing-b", type=Path, default=Path("data/missing_accessions_TEDseqB.txt")
    )
    parser.add_argument("--cleaned-dir", type=Path, default=Path("outputs/cleaned"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--max-violin-samples",
        type=int,
        default=50000,
        help="Max samples per group for violin plot (for plotting speed only).",
    )
    parser.add_argument("--tree-top-classes", type=int, default=6)
    parser.add_argument("--tree-top-arch", type=int, default=4)
    parser.add_argument("--tree-top-topo", type=int, default=3)
    parser.add_argument("--tree-top-sf", type=int, default=2)
    parser.add_argument("--circular-max-leaves", type=int, default=60)
    parser.add_argument("--circular-label-top", type=int, default=30)
    parser.add_argument(
        "--circular-leaf-level",
        choices=["topology", "superfamily"],
        default="topology",
        help="Legacy option retained for compatibility; both topology and superfamily circular figures are generated.",
    )
    parser.add_argument("--circular-min-label-angle-deg", type=float, default=8.0)
    parser.add_argument("--circular-label-target", type=int, default=30)
    parser.add_argument("--circular-label-max-candidates", type=int, default=60)
    parser.add_argument("--coverage-sample-n", type=int, default=30000)
    parser.add_argument("--coverage-max-pos", type=int, default=2048)
    parser.add_argument("--coverage-bin-size", type=int, default=256)
    return parser.parse_args()


def configure_plot_style() -> None:
    """Apply a clean static style suitable for papers."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Ensure output directories exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, figures_dir: Path, dpi: int) -> None:
    """Save one figure as PNG and PDF with consistent settings."""
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_empty_figure(title: str, message: str) -> plt.Figure:
    """Create a placeholder figure when there is no data to plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return fig


def load_missing_accessions(paths: Sequence[Path]) -> set[str]:
    """Load missing accessions from one or more TSV files."""
    all_accessions: set[str] = set()
    for path in paths:
        print(f"[LOAD] Missing accession file: {path}")
        df = pd.read_csv(path, sep="\t", dtype=str)
        if df.empty:
            continue

        accession_col = None
        for col in df.columns:
            if col.strip().lower() == "accession":
                accession_col = col
                break
        if accession_col is None:
            accession_col = df.columns[0]
            print(
                f"[WARN] Could not find 'accession' column in {path.name}; "
                f"using first column '{accession_col}'."
            )

        cleaned = (
            df[accession_col]
            .astype("string")
            .str.strip()
            .dropna()
        )
        cleaned = cleaned[cleaned != ""]
        all_accessions.update(cleaned.tolist())
    print(f"[INFO] Total unique missing accessions loaded: {len(all_accessions):,}")
    return all_accessions


def load_chains_csv(path: Path) -> pd.DataFrame:
    """Load chains CSV and enforce required columns."""
    print(f"[LOAD] Chains CSV: {path}")
    df = pd.read_csv(path, dtype={"uniprot_id": "string", "chopping_star": "string"})
    required = {"uniprot_id", "chopping_star"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required column(s): {sorted(missing_cols)}")

    df["uniprot_id"] = df["uniprot_id"].astype("string").str.strip()
    df["chopping_star"] = df["chopping_star"].astype("string")
    return df


def split_domain_tokens(chopping_star: Optional[str]) -> List[str]:
    """Split chopping_star into domain tokens (one token per domain)."""
    if chopping_star is None or pd.isna(chopping_star):
        return []
    text = str(chopping_star).strip()
    if text == "":
        return []
    tokens = [token.strip() for token in DOMAIN_SPLIT_RE.split(text) if token.strip()]
    return tokens


def parse_cath_label(cath_label_raw: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse CATH label into up to four hierarchy levels.

    Accepts labels with 2, 3, or 4 dot-separated numeric levels.
    Nonstandard labels are preserved in raw form and return Nones in parsed levels.
    """
    label = cath_label_raw.strip()
    parts = [part.strip() for part in label.split(".")]
    if 2 <= len(parts) <= 4 and all(part.isdigit() for part in parts):
        padded = parts + [None] * (4 - len(parts))
        return padded[0], padded[1], padded[2], padded[3]
    return None, None, None, None


def _parse_ranges(
    ranges_raw: str,
) -> Tuple[bool, Optional[str], int, bool, str, Optional[int], Optional[int]]:
    """
    Parse range string like '2-143_287-314' into summary metrics.

    Returns:
        parse_ok, parse_error, segments_count, is_non_contiguous,
        segment_lengths, domain_length, start_min, end_max
    """
    segments = [seg.strip() for seg in ranges_raw.split("_") if seg.strip()]
    if not segments:
        return False, "empty_ranges", 0, False, "", None, None, None

    starts: List[int] = []
    ends: List[int] = []
    lengths: List[int] = []
    for segment in segments:
        match = SEGMENT_RE.match(segment)
        if not match:
            return (
                False,
                f"malformed_segment:{segment}",
                0,
                False,
                "",
                None,
                None,
                None,
            )
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            return (
                False,
                f"range_end_before_start:{segment}",
                0,
                False,
                "",
                None,
                None,
                None,
            )
        starts.append(start)
        ends.append(end)
        lengths.append(end - start + 1)  # inclusive residue range

    segments_count = len(lengths)
    segment_lengths = ";".join(str(length) for length in lengths)
    return (
        True,
        None,
        segments_count,
        segments_count > 1,
        segment_lengths,
        int(sum(lengths)),
        int(min(starts)),
        int(max(ends)),
    )


def parse_domains(filtered_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse chopping_star into one row per domain and collect parsing issues."""
    print("[PARSE] Parsing chopping_star into domain-level records...")
    domain_records: List[Dict[str, object]] = []
    parsing_issues: List[Dict[str, object]] = []

    for row_i, (uniprot_id, chopping_star) in enumerate(
        filtered_df[["uniprot_id", "chopping_star"]].itertuples(index=False, name=None),
        start=1,
    ):
        if row_i % 50000 == 0:
            print(f"[PARSE] Processed {row_i:,} chain rows...")

        tokens = split_domain_tokens(chopping_star)
        if not tokens:
            parsing_issues.append(
                {
                    "uniprot_id": uniprot_id,
                    "domain_index": None,
                    "domain_raw": "",
                    "chopping_star_raw": chopping_star,
                    "parse_error": "empty_or_missing_chopping_star",
                }
            )
            continue

        for domain_index, token in enumerate(tokens, start=1):
            record: Dict[str, object] = {
                "uniprot_id": uniprot_id,
                "domain_index": domain_index,
                "domain_raw": token,
                "ranges_raw": "",
                "cath_label_raw": "",
                "cath_assigned": False,
                "segments_count": 0,
                "is_non_contiguous": False,
                "segment_lengths": "",
                "domain_length": np.nan,
                "start_min": np.nan,
                "end_max": np.nan,
                "cath_class": None,
                "cath_architecture": None,
                "cath_topology": None,
                "cath_homologous_superfamily": None,
                "parse_ok": False,
                "parse_error": "",
            }

            if "|" not in token:
                record["parse_error"] = "missing_pipe_separator"
                domain_records.append(record)
                parsing_issues.append(
                    {
                        "uniprot_id": uniprot_id,
                        "domain_index": domain_index,
                        "domain_raw": token,
                        "chopping_star_raw": chopping_star,
                        "parse_error": record["parse_error"],
                    }
                )
                continue

            left, right = token.split("|", 1)
            ranges_raw = left.strip()
            cath_raw = right.strip()
            record["ranges_raw"] = ranges_raw
            record["cath_label_raw"] = cath_raw
            record["cath_assigned"] = bool(cath_raw not in {"", "-"})

            (
                ranges_ok,
                ranges_error,
                segments_count,
                is_non_contiguous,
                segment_lengths,
                domain_length,
                start_min,
                end_max,
            ) = _parse_ranges(ranges_raw)

            if not ranges_ok:
                record["parse_error"] = ranges_error or "range_parse_failed"
                domain_records.append(record)
                parsing_issues.append(
                    {
                        "uniprot_id": uniprot_id,
                        "domain_index": domain_index,
                        "domain_raw": token,
                        "chopping_star_raw": chopping_star,
                        "parse_error": record["parse_error"],
                    }
                )
                continue

            record["segments_count"] = segments_count
            record["is_non_contiguous"] = is_non_contiguous
            record["segment_lengths"] = segment_lengths
            record["domain_length"] = domain_length
            record["start_min"] = start_min
            record["end_max"] = end_max

            if record["cath_assigned"]:
                c_class, c_arch, c_top, c_sf = parse_cath_label(cath_raw)
                record["cath_class"] = c_class
                record["cath_architecture"] = c_arch
                record["cath_topology"] = c_top
                record["cath_homologous_superfamily"] = c_sf

            record["parse_ok"] = True
            record["parse_error"] = ""
            domain_records.append(record)

    domains_df = pd.DataFrame(domain_records)
    issues_df = pd.DataFrame(parsing_issues)

    if domains_df.empty:
        domains_df = pd.DataFrame(
            columns=[
                "uniprot_id",
                "domain_index",
                "domain_raw",
                "ranges_raw",
                "cath_label_raw",
                "cath_assigned",
                "segments_count",
                "is_non_contiguous",
                "segment_lengths",
                "domain_length",
                "start_min",
                "end_max",
                "cath_class",
                "cath_architecture",
                "cath_topology",
                "cath_homologous_superfamily",
                "parse_ok",
                "parse_error",
            ]
        )
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=[
                "uniprot_id",
                "domain_index",
                "domain_raw",
                "chopping_star_raw",
                "parse_error",
            ]
        )
    return domains_df, issues_df


def find_duplicate_uniprot_ids(chains_df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate uniprot_ids with occurrence counts."""
    counts = chains_df["uniprot_id"].value_counts(dropna=False)
    dups = counts[counts > 1].rename_axis("uniprot_id").reset_index(name="row_count")
    return dups


def build_chain_summary(filtered_df: pd.DataFrame, domains_df: pd.DataFrame) -> pd.DataFrame:
    """Build chain-level summary table from parsed domain records."""
    print("[SUM] Building chain summary table...")
    unique_uniprot = (
        filtered_df[["uniprot_id"]]
        .drop_duplicates()
        .copy()
    )

    if domains_df.empty:
        summary = unique_uniprot.copy()
        summary["n_domains"] = 0
        summary["n_assigned_domains"] = 0
        summary["n_unassigned_domains"] = 0
        summary["n_non_contiguous_domains"] = 0
        summary["has_non_contiguous_domain"] = False
        summary["annotated_domain_length_sum"] = 0
        summary["n_domains_parse_failed"] = 0
        return summary

    grouped = domains_df.groupby("uniprot_id", dropna=False)
    summary = grouped.agg(
        n_domains=("domain_index", "size"),
        n_assigned_domains=("cath_assigned", lambda s: int(pd.Series(s).fillna(False).sum())),
        n_non_contiguous_domains=(
            "is_non_contiguous",
            lambda s: int(pd.Series(s).fillna(False).sum()),
        ),
        annotated_domain_length_sum=(
            "domain_length",
            lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        ),
        n_domains_parse_failed=("parse_ok", lambda s: int((~pd.Series(s).fillna(False)).sum())),
    ).reset_index()
    summary["n_unassigned_domains"] = summary["n_domains"] - summary["n_assigned_domains"]
    summary["has_non_contiguous_domain"] = summary["n_non_contiguous_domains"] > 0

    merged = unique_uniprot.merge(summary, on="uniprot_id", how="left")
    int_cols = [
        "n_domains",
        "n_assigned_domains",
        "n_unassigned_domains",
        "n_non_contiguous_domains",
        "annotated_domain_length_sum",
        "n_domains_parse_failed",
    ]
    for col in int_cols:
        merged[col] = merged[col].fillna(0).astype(int)
    merged["has_non_contiguous_domain"] = merged["has_non_contiguous_domain"].fillna(False).astype(bool)

    domain_counts = domains_df.groupby("uniprot_id", dropna=False).size().rename("domain_count_check")
    check_df = merged.merge(
        domain_counts.reset_index(),
        on="uniprot_id",
        how="left",
    )
    check_df["domain_count_check"] = check_df["domain_count_check"].fillna(0).astype(int)
    if not (check_df["n_domains"] == check_df["domain_count_check"]).all():
        mismatch = int((check_df["n_domains"] != check_df["domain_count_check"]).sum())
        raise AssertionError(
            f"Validation failed: n_domains mismatch for {mismatch} uniprot_id rows."
        )

    return merged


def _auto_hist_bins(values: np.ndarray, max_bins: int = 100) -> int:
    """Choose a sensible histogram bin count."""
    if values.size <= 1:
        return 1
    unique_count = np.unique(values).size
    return int(max(10, min(max_bins, unique_count)))


def _prep_assigned_domains(domains_df: pd.DataFrame) -> pd.DataFrame:
    """Subset to assigned + parse_ok domains for CATH analyses."""
    subset = domains_df[(domains_df["parse_ok"]) & (domains_df["cath_assigned"])].copy()
    return subset


def _build_cath_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build reusable human-readable CATH level labels."""
    out = df.copy()
    out["cath_class_label"] = out["cath_class"]

    out["cath_architecture_label"] = np.where(
        out["cath_class"].notna() & out["cath_architecture"].notna(),
        out["cath_class"].astype(str) + "." + out["cath_architecture"].astype(str),
        None,
    )
    out["cath_topology_label"] = np.where(
        out["cath_architecture_label"].notna() & out["cath_topology"].notna(),
        out["cath_architecture_label"].astype(str) + "." + out["cath_topology"].astype(str),
        None,
    )
    out["cath_superfamily_label"] = np.where(
        out["cath_topology_label"].notna() & out["cath_homologous_superfamily"].notna(),
        out["cath_topology_label"].astype(str) + "." + out["cath_homologous_superfamily"].astype(str),
        None,
    )
    return out


def _get_contiguity_length_arrays(domains_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return numeric domain-length arrays for contiguous and non-contiguous domains."""
    plot_df = domains_df.copy()
    plot_df["domain_length_num"] = pd.to_numeric(plot_df["domain_length"], errors="coerce")
    plot_df = plot_df[(plot_df["parse_ok"]) & (plot_df["domain_length_num"] > 0)]
    contig = plot_df.loc[~plot_df["is_non_contiguous"], "domain_length_num"].to_numpy(dtype=float)
    non_contig = plot_df.loc[plot_df["is_non_contiguous"], "domain_length_num"].to_numpy(dtype=float)
    return contig, non_contig


def _draw_domain_length_box_on_ax(
    ax: plt.Axes,
    contig: np.ndarray,
    non_contig: np.ndarray,
    title: str = "Domain Length: Contiguous vs Non-contiguous Boxplot",
    positions: Tuple[float, float] = (1.0, 1.35),
) -> None:
    """Draw domain-length boxplot for contiguous vs non-contiguous domains on a given axis."""
    if contig.size == 0 and non_contig.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No parseable domain lengths available.", ha="center", va="center")
        return

    data = [contig if contig.size else np.array([np.nan]), non_contig if non_contig.size else np.array([np.nan])]
    ax.boxplot(
        data,
        positions=list(positions),
        widths=0.22,
        showfliers=False,
        whis=(5, 95),
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(["Contiguous", "Non-contiguous"])
    ax.set_xlim(min(positions) - 0.22, max(positions) + 0.22)
    ax.set_title(title)
    ax.set_ylabel("Domain Length (residues)")


def _draw_domain_length_violin_on_ax(
    ax: plt.Axes,
    contig: np.ndarray,
    non_contig: np.ndarray,
    max_violin_samples: int,
    title: str = "Domain Length: Contiguous vs Non-contiguous",
    positions: Tuple[float, float] = (1.0, 1.35),
) -> None:
    """Draw domain-length violin plot on a given axis using deterministic subsampling."""
    if contig.size == 0 and non_contig.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No parseable domain lengths available.", ha="center", va="center")
        return

    rng = np.random.default_rng(RNG_SEED)
    contig_violin = contig
    non_contig_violin = non_contig
    if contig_violin.size > max_violin_samples:
        idx = rng.choice(contig_violin.size, size=max_violin_samples, replace=False)
        contig_violin = contig_violin[idx]
    if non_contig_violin.size > max_violin_samples:
        idx = rng.choice(non_contig_violin.size, size=max_violin_samples, replace=False)
        non_contig_violin = non_contig_violin[idx]

    violin_data = [
        contig_violin if contig_violin.size else np.array([np.nan]),
        non_contig_violin if non_contig_violin.size else np.array([np.nan]),
    ]
    v = ax.violinplot(
        violin_data,
        positions=list(positions),
        widths=0.20,
        showmeans=True,
        showextrema=False,
    )
    for body in v["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(["Contiguous", "Non-contiguous"])
    ax.set_xlim(min(positions) - 0.22, max(positions) + 0.22)
    ax.set_title(title)
    ax.set_ylabel("Domain Length (residues)")


def _draw_top_barh_on_ax(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int = 15,
    color: str = "#277DA1",
    gradient: bool = False,
    colormap: str = "cividis",
) -> None:
    """Draw top-N horizontal bar chart onto a provided axis."""
    counts = series.dropna().astype(str).value_counts().head(top_n)
    if counts.empty:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No assigned CATH labels available.", ha="center", va="center")
        return

    counts_sorted = counts.sort_values(ascending=True)
    if gradient:
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0.12, 0.92, len(counts_sorted)))
    else:
        colors = color
    ax.barh(
        counts_sorted.index,
        counts_sorted.values,
        color=colors,
        alpha=0.9,
        edgecolor="#2F2F2F" if gradient else None,
        linewidth=0.25 if gradient else 0.0,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _draw_domains_per_chain_hist_on_ax(
    ax: plt.Axes,
    chain_summary_df: pd.DataFrame,
    title: str,
    color: str,
    zoom_percentile: Optional[float] = None,
) -> None:
    """Draw domains-per-chain bars with gradient color, optionally percentile-clipped."""
    values = pd.to_numeric(chain_summary_df["n_domains"], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[values > 0]  # remove n_domains == 0 for this visualization
    if values.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center")
        return

    plot_vals = values
    if zoom_percentile is not None:
        cutoff = int(np.ceil(np.percentile(values, float(zoom_percentile))))
        plot_vals = values[values <= cutoff]
    if plot_vals.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center")
        return

    x_vals, counts = np.unique(plot_vals.astype(int), return_counts=True)
    cmap = plt.get_cmap("cividis")
    colors = cmap(np.linspace(0.12, 0.92, len(x_vals)))
    ax.bar(
        x_vals,
        counts,
        width=0.88,
        color=colors,
        edgecolor="#2F2F2F",
        linewidth=0.25,
    )
    ax.set_title(title)
    ax.set_xlabel("Domains per Chain")
    ax.set_ylabel("Number of Chains")
    ax.set_xlim(float(x_vals.min()) - 0.7, float(x_vals.max()) + 0.7)


def plot_domains_per_chain(chain_summary_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot histogram of number of domains per chain (full + zoom)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_domains_per_chain_hist_on_ax(
        ax=ax,
        chain_summary_df=chain_summary_df,
        title="Domains per Chain Full Range",
        color="#2E86AB",
        zoom_percentile=None,
    )
    save_figure(fig, "domains_per_chain_hist", figures_dir, dpi)

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_domains_per_chain_hist_on_ax(
        ax=ax,
        chain_summary_df=chain_summary_df,
        title="Domains per Chain",
        color="#1B998B",
        zoom_percentile=99.5,
    )
    save_figure(fig, "domains_per_chain_hist_zoom", figures_dir, dpi)


def plot_domain_length_distributions(domains_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot regular and log-x histograms for domain lengths."""
    lengths = pd.to_numeric(domains_df["domain_length"], errors="coerce")
    lengths = lengths.dropna()
    lengths = lengths[lengths > 0]
    if lengths.empty:
        fig = make_empty_figure("Distribution of Domain Lengths", "No parseable domain lengths available.")
        save_figure(fig, "domain_length_hist", figures_dir, dpi)
        fig = make_empty_figure(
            "Distribution of Domain Lengths (Log X-axis)",
            "No parseable domain lengths available.",
        )
        save_figure(fig, "domain_length_hist_logx", figures_dir, dpi)
        return

    vals = lengths.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(vals, bins=80, color="#F46036", alpha=0.9, edgecolor="black", linewidth=0.2)
    ax.set_title("Distribution of Domain Lengths")
    ax.set_xlabel("Domain Length (residues)")
    ax.set_ylabel("Number of Domains")
    save_figure(fig, "domain_length_hist", figures_dir, dpi)

    xmin, xmax = float(vals.min()), float(vals.max())
    if xmin <= 0:
        xmin = 1.0
    bins = np.logspace(np.log10(xmin), np.log10(max(xmax, xmin + 1.0)), 80)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(vals, bins=bins, color="#E59F71", alpha=0.9, edgecolor="black", linewidth=0.2)
    ax.set_xscale("log")
    ax.set_title("Distribution of Domain Lengths (Log-scaled X-axis)")
    ax.set_xlabel("Domain Length (residues, log scale)")
    ax.set_ylabel("Number of Domains")
    save_figure(fig, "domain_length_hist_logx", figures_dir, dpi)


def _infer_chain_seq_len_est(
    filtered_df: pd.DataFrame, domains_df: pd.DataFrame
) -> Tuple[pd.Series, str]:
    """Infer per-chain sequence-length estimate from chain columns or parsed max end position."""
    lower_to_col = {col.strip().lower(): col for col in filtered_df.columns}
    candidate_cols = [
        "chain_length",
        "seq_length",
        "sequence_length",
        "length",
        "seqlen",
        "protein_length",
    ]
    for key in candidate_cols:
        if key in lower_to_col:
            raw_col = lower_to_col[key]
            candidate_df = filtered_df[["uniprot_id", raw_col]].copy()
            candidate_df["seq_len_est"] = pd.to_numeric(candidate_df[raw_col], errors="coerce")
            candidate_df = candidate_df[candidate_df["seq_len_est"].notna() & (candidate_df["seq_len_est"] > 0)]
            if not candidate_df.empty:
                series = candidate_df.groupby("uniprot_id", dropna=False)["seq_len_est"].max()
                return series, f"column:{raw_col}"

    if domains_df.empty:
        return pd.Series(dtype=float), "none"

    parse_ok = domains_df[domains_df["parse_ok"]].copy()
    if parse_ok.empty:
        return pd.Series(dtype=float), "none"

    parse_ok["end_max_num"] = pd.to_numeric(parse_ok["end_max"], errors="coerce")
    parse_ok = parse_ok[parse_ok["end_max_num"].notna() & (parse_ok["end_max_num"] > 0)]
    if parse_ok.empty:
        return pd.Series(dtype=float), "none"

    series = (
        parse_ok.groupby("uniprot_id", dropna=False)["end_max_num"]
        .max()
        .rename("seq_len_est")
    )
    return series, "derived:max_domain_end"


def _infer_sequence_lengths_for_overlay(
    filtered_df: pd.DataFrame, domains_df: pd.DataFrame
) -> Tuple[np.ndarray, str]:
    """Infer sequence lengths from chain columns, then fallback to max parsed end_max per chain."""
    seq_len_series, source = _infer_chain_seq_len_est(filtered_df, domains_df)
    if seq_len_series.empty:
        return np.array([], dtype=float), "none"

    mapped = filtered_df[["uniprot_id"]].merge(
        seq_len_series.rename("seq_len_est").reset_index(),
        on="uniprot_id",
        how="left",
    )
    seq_lengths = pd.to_numeric(mapped["seq_len_est"], errors="coerce")
    seq_lengths = seq_lengths[(seq_lengths > 0) & seq_lengths.notna()]
    if seq_lengths.empty:
        return np.array([], dtype=float), "none"
    return seq_lengths.to_numpy(dtype=float), source


def _prepare_domain_vs_sequence_lengths(
    filtered_df: pd.DataFrame, domains_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Prepare domain and sequence length arrays for overlay plotting."""
    domain_lengths = pd.to_numeric(domains_df["domain_length"], errors="coerce")
    domain_lengths = domain_lengths[(domain_lengths > 0) & domain_lengths.notna()].to_numpy(dtype=float)
    seq_lengths, source = _infer_sequence_lengths_for_overlay(filtered_df, domains_df)
    return domain_lengths, seq_lengths, source


def _draw_domain_vs_sequence_overlay_on_ax(
    ax: plt.Axes,
    domain_lengths: np.ndarray,
    sequence_lengths: np.ndarray,
    title: str = "Domain vs Sequence Length Distribution",
) -> None:
    """Draw domain-vs-sequence length overlay on a provided axis."""
    if domain_lengths.size == 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No parseable domain lengths available.", ha="center", va="center")
        return

    if sequence_lengths.size == 0:
        bins = np.histogram_bin_edges(domain_lengths, bins=80)
        ax.hist(
            domain_lengths,
            bins=bins,
            color="#F46036",
            alpha=0.82,
            edgecolor="#2F2F2F",
            linewidth=0.2,
            label="Domain length",
        )
        ax.set_title(title)
        ax.set_xlabel("Length (aa)")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        return

    combined = np.concatenate([domain_lengths, sequence_lengths])
    bins = np.histogram_bin_edges(combined, bins=80)
    ax.hist(
        domain_lengths,
        bins=bins,
        color="#F46036",
        alpha=0.58,
        edgecolor="#2F2F2F",
        linewidth=0.2,
        label="Domain length",
    )
    ax.hist(
        sequence_lengths,
        bins=bins,
        color="#277DA1",
        alpha=0.50,
        edgecolor="#2F2F2F",
        linewidth=0.2,
        label="Sequence length",
    )
    ax.set_title(title)
    ax.set_xlabel("Length (aa)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)


def plot_domain_vs_sequence_length_overlay(
    filtered_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    """Overlay domain-length and sequence-length distributions in one linear-scale figure."""
    domain_lengths, seq_lengths, source = _prepare_domain_vs_sequence_lengths(filtered_df, domains_df)
    if domain_lengths.size == 0:
        fig = make_empty_figure("Domain vs Sequence Length Distribution", "No parseable domain lengths available.")
        save_figure(fig, "domain_vs_sequence_length_overlay", figures_dir, dpi)
        return

    if seq_lengths.size == 0:
        print(
            "[WARN] Could not infer sequence lengths for overlay plot. "
            "Saving domain-only distribution as domain_vs_sequence_length_overlay."
        )
    else:
        print(f"[INFO] Sequence length overlay source: {source}")

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_domain_vs_sequence_overlay_on_ax(
        ax=ax,
        domain_lengths=domain_lengths,
        sequence_lengths=seq_lengths,
        title="Domain vs Sequence Length Distribution",
    )
    save_figure(fig, "domain_vs_sequence_length_overlay", figures_dir, dpi)


def compute_domain_coverage_and_heatmap(
    filtered_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    cleaned_dir: Path,
    sample_n: int,
    max_pos: int,
    bin_size: int,
    rng_seed: int = RNG_SEED,
) -> Dict[str, object]:
    """
    Compute and save domain coverage vector + heatmap matrix for positions 1..max_pos.

    Uses a deterministic subsample of unique uniprot_id values for runtime control.
    """
    max_pos = max(1, int(max_pos))
    bin_size = max(1, int(bin_size))
    sample_n = max(1, int(sample_n))

    valid_ids = filtered_df["uniprot_id"].astype("string").str.strip()
    valid_ids = valid_ids[valid_ids.notna() & (valid_ids != "")]
    unique_ids = np.array(sorted(valid_ids.unique().tolist()), dtype=object)

    if unique_ids.size == 0:
        coverage_vector = np.zeros(max_pos, dtype=float)
        n_bins = int(np.ceil(max_pos / bin_size))
        heatmap_matrix = np.full((n_bins, max_pos), np.nan, dtype=float)
        bin_labels = [
            f"{i * bin_size + 1}-{min((i + 1) * bin_size, max_pos)}"
            for i in range(n_bins)
        ]
        bin_counts = np.zeros(n_bins, dtype=int)
        vector_df = pd.DataFrame(
            {"position": np.arange(1, max_pos + 1, dtype=int), "mean_coverage": coverage_vector}
        )
        matrix_df = pd.DataFrame(
            heatmap_matrix,
            columns=[str(i) for i in range(1, max_pos + 1)],
        )
        bins_df = pd.DataFrame(
            {
                "bin_index": np.arange(1, n_bins + 1, dtype=int),
                "bin_label": bin_labels,
                "n_chains_in_bin": bin_counts,
            }
        )
        vector_df.to_csv(cleaned_dir / "domain_coverage_vector_1to2048.csv", index=False)
        matrix_df.to_csv(cleaned_dir / "domain_coverage_heatmap_matrix.csv", index=False)
        bins_df.to_csv(cleaned_dir / "domain_coverage_heatmap_bins.csv", index=False)
        return {
            "coverage_vector_df": vector_df,
            "heatmap_matrix": heatmap_matrix,
            "bin_labels": bin_labels,
            "bin_counts": bin_counts,
            "sample_size": 0,
            "max_pos": max_pos,
            "n_bins": n_bins,
            "bin_size": bin_size,
            "seq_len_source": "none",
        }

    rng = np.random.default_rng(rng_seed)
    if unique_ids.size > sample_n:
        sampled_ids = rng.choice(unique_ids, size=sample_n, replace=False)
    else:
        sampled_ids = unique_ids
    sampled_ids = np.array(sorted(sampled_ids.tolist()), dtype=object)
    sampled_set = set(sampled_ids.tolist())

    seq_len_series, seq_source = _infer_chain_seq_len_est(filtered_df, domains_df)
    seq_map = pd.Series(sampled_ids, name="uniprot_id").to_frame()
    if not seq_len_series.empty:
        seq_map = seq_map.merge(
            seq_len_series.rename("seq_len_est").reset_index(),
            on="uniprot_id",
            how="left",
        )
    else:
        seq_map["seq_len_est"] = np.nan
    seq_values = pd.to_numeric(seq_map["seq_len_est"], errors="coerce").to_numpy(dtype=float)

    n_bins = int(np.ceil(max_pos / bin_size))
    global_covered = np.zeros(max_pos, dtype=np.int64)
    heatmap_counts = np.zeros((n_bins, max_pos), dtype=np.int64)
    bin_chain_counts = np.zeros(n_bins, dtype=np.int64)

    parse_ok = _to_bool_series(domains_df["parse_ok"]) if "parse_ok" in domains_df.columns else pd.Series(
        False, index=domains_df.index
    )
    ranges_df = domains_df.loc[parse_ok, ["uniprot_id", "ranges_raw"]].copy()
    ranges_df["uniprot_id"] = ranges_df["uniprot_id"].astype("string")
    ranges_df = ranges_df[ranges_df["uniprot_id"].isin(sampled_set)]

    chain_segments: Dict[str, List[Tuple[int, int]]] = {}
    for uniprot_id, ranges_raw in ranges_df.itertuples(index=False, name=None):
        if pd.isna(ranges_raw):
            continue
        uid = str(uniprot_id)
        segs = [seg.strip() for seg in str(ranges_raw).split("_") if seg.strip()]
        for seg in segs:
            match = SEGMENT_RE.match(seg)
            if not match:
                continue
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start or end < 1 or start > max_pos:
                continue
            start = max(1, start)
            end = min(max_pos, end)
            if end < start:
                continue
            chain_segments.setdefault(uid, []).append((start, end))

    for i, uid in enumerate(sampled_ids):
        chain_diff = np.zeros(max_pos + 2, dtype=np.int16)
        for start, end in chain_segments.get(str(uid), []):
            chain_diff[start] += 1
            chain_diff[end + 1] -= 1
        covered = np.cumsum(chain_diff)[1 : max_pos + 1] > 0
        covered_int = covered.astype(np.int64)
        global_covered += covered_int

        seq_len = seq_values[i] if i < len(seq_values) else np.nan
        if pd.notna(seq_len) and seq_len > 0:
            seq_len_clip = int(min(max_pos, max(1, int(np.floor(seq_len)))))
            bin_idx = (seq_len_clip - 1) // bin_size
            if 0 <= bin_idx < n_bins:
                heatmap_counts[bin_idx] += covered_int
                bin_chain_counts[bin_idx] += 1

    sample_size = int(len(sampled_ids))
    mean_coverage = (
        global_covered.astype(float) / float(sample_size)
        if sample_size > 0
        else np.zeros(max_pos, dtype=float)
    )
    heatmap_matrix = np.full((n_bins, max_pos), np.nan, dtype=float)
    for b in range(n_bins):
        if bin_chain_counts[b] > 0:
            heatmap_matrix[b, :] = heatmap_counts[b, :] / float(bin_chain_counts[b])

    positions = np.arange(1, max_pos + 1, dtype=int)
    coverage_vector_df = pd.DataFrame(
        {
            "position": positions,
            "mean_coverage": mean_coverage,
        }
    )
    heatmap_matrix_df = pd.DataFrame(
        heatmap_matrix,
        columns=[str(pos) for pos in positions],
    )

    bin_labels = []
    for b in range(n_bins):
        start = b * bin_size + 1
        end = min((b + 1) * bin_size, max_pos)
        bin_labels.append(f"{start}-{end}")

    bins_df = pd.DataFrame(
        {
            "bin_index": np.arange(1, n_bins + 1, dtype=int),
            "bin_label": bin_labels,
            "n_chains_in_bin": bin_chain_counts.astype(int),
        }
    )

    coverage_vector_df.to_csv(cleaned_dir / "domain_coverage_vector_1to2048.csv", index=False)
    heatmap_matrix_df.to_csv(cleaned_dir / "domain_coverage_heatmap_matrix.csv", index=False)
    bins_df.to_csv(cleaned_dir / "domain_coverage_heatmap_bins.csv", index=False)

    return {
        "coverage_vector_df": coverage_vector_df,
        "heatmap_matrix": heatmap_matrix,
        "bin_labels": bin_labels,
        "bin_counts": bin_chain_counts.astype(int),
        "sample_size": sample_size,
        "max_pos": max_pos,
        "n_bins": n_bins,
        "bin_size": bin_size,
        "seq_len_source": seq_source,
    }


def plot_domain_coverage_distribution(
    coverage_vector_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    max_pos: int,
) -> None:
    """Plot mean domain coverage across sequence positions 1..max_pos."""
    if coverage_vector_df.empty:
        fig = make_empty_figure("Domain Coverage Along Sequence Position", "No coverage data available.")
        save_figure(fig, "domain_coverage_distribution_1to2048", figures_dir, dpi)
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = pd.to_numeric(coverage_vector_df["position"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(coverage_vector_df["mean_coverage"], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y, color="#1D4E89", linewidth=2.0)
    ax.fill_between(x, y, 0.0, color="#1D4E89", alpha=0.18)
    ax.set_xlim(1, max_pos)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Domain Coverage Along Sequence Position")
    ax.set_xlabel("Sequence position aa")
    ax.set_ylabel("Fraction of sequences covered by a domain")
    save_figure(fig, "domain_coverage_distribution_1to2048", figures_dir, dpi)


def plot_domain_coverage_heatmap(
    heatmap_matrix: np.ndarray,
    bin_labels: Sequence[str],
    figures_dir: Path,
    dpi: int,
    max_pos: int,
    bin_size: int,
) -> None:
    """Plot domain coverage heatmap by sequence-length bins."""
    if heatmap_matrix.size == 0 or len(bin_labels) == 0:
        fig = make_empty_figure("Domain Coverage Heatmap by Sequence Length", "No coverage data available.")
        save_figure(fig, "domain_coverage_heatmap_1to2048", figures_dir, dpi)
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#F2F2F2")
    masked = np.ma.masked_invalid(heatmap_matrix)
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[1, max_pos, 0.5, len(bin_labels) + 0.5],
    )

    step = max(1, int(bin_size))
    x_ticks = list(range(1, max_pos + 1, step))
    if x_ticks[-1] != max_pos:
        x_ticks.append(max_pos)
    ax.set_xticks(x_ticks)
    ax.set_yticks(np.arange(1, len(bin_labels) + 1))
    ax.set_yticklabels(list(bin_labels))
    ax.set_title("Domain Coverage Heatmap by Sequence Length")
    ax.set_xlabel("Sequence position aa")
    ax.set_ylabel("Sequence length bin aa")

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Fraction covered")
    save_figure(fig, "domain_coverage_heatmap_1to2048", figures_dir, dpi)


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed-type series into a boolean series deterministically."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) != 0.0
    normalized = series.astype("string").str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    return normalized.isin(true_values)


def _format_count(value: float) -> str:
    """Format integer-like counts with thousands separators."""
    return f"{int(round(float(value))):,}"


def _format_float(value: float, decimals: int = 2) -> str:
    """Format floating-point values with thousands separators."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _format_percent(value: float, decimals: int = 2) -> str:
    """Format fraction values as percentages."""
    if pd.isna(value):
        return "N/A"
    return f"{100.0 * float(value):.{decimals}f}%"


def _detect_training_chains_count(filtered_df: pd.DataFrame) -> str:
    """Detect training-chain count from likely split columns, if available."""
    if filtered_df.empty or "uniprot_id" not in filtered_df.columns:
        return "N/A (no split column)"

    prioritized_cols = [
        col
        for col in filtered_df.columns
        if any(token in col.strip().lower() for token in ["split", "partition", "subset", "fold", "set", "train"])
    ]
    fallback_cols = [col for col in filtered_df.columns if col not in prioritized_cols]
    candidate_cols = prioritized_cols + fallback_cols

    train_regex = re.compile(r"\btrain(?:ing)?\b", re.IGNORECASE)
    for col in candidate_cols:
        series = filtered_df[col]
        if series.isna().all():
            continue

        col_lower = col.strip().lower()
        normalized = series.astype("string").str.strip().str.lower()
        if normalized.isna().all():
            continue

        if "train" in col_lower:
            truthy = normalized.isin({"1", "true", "t", "yes", "y", "train", "training"})
            if truthy.any():
                return _format_count(filtered_df.loc[truthy, "uniprot_id"].nunique())

        match_mask = normalized.str.contains(train_regex, na=False)
        if match_mask.any():
            return _format_count(filtered_df.loc[match_mask, "uniprot_id"].nunique())

    return "N/A (no split column)"


def _build_poster_stats_dataframe(
    filtered_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    chain_summary_df: pd.DataFrame,
    eda_summary: Dict[str, object],
) -> pd.DataFrame:
    """Build a compact poster-friendly stats table."""
    valid_chains = filtered_df["uniprot_id"].astype("string").str.strip()
    valid_chains = valid_chains[valid_chains.notna() & (valid_chains != "")]
    valid_chain_count = int(valid_chains.nunique())

    parsed_domains_count = int(eda_summary.get("parsed_domains_count", len(domains_df)))
    parse_failures = int(eda_summary.get("domain_parse_failures_count", 0))
    parse_success_rate = (1.0 - (parse_failures / parsed_domains_count)) if parsed_domains_count > 0 else np.nan
    assigned_domains_count = int(eda_summary.get("assigned_domains_count", 0))
    unassigned_domains_count = int(eda_summary.get("unassigned_domains_count", 0))
    assigned_fraction = float(eda_summary.get("assigned_fraction", np.nan))

    n_domains = pd.to_numeric(chain_summary_df.get("n_domains", pd.Series(dtype=float)), errors="coerce").dropna()
    n_domains_nonzero = n_domains[n_domains > 0]
    mean_domains_per_chain = n_domains_nonzero.mean() if not n_domains_nonzero.empty else np.nan
    median_domains_per_chain = n_domains_nonzero.median() if not n_domains_nonzero.empty else np.nan

    has_non_contig = (
        _to_bool_series(chain_summary_df["has_non_contiguous_domain"])
        if "has_non_contiguous_domain" in chain_summary_df.columns
        else pd.Series(dtype=bool)
    )
    frac_chains_with_non_contig = has_non_contig.mean() if not has_non_contig.empty else np.nan
    total_non_contig_domains = pd.to_numeric(
        chain_summary_df.get("n_non_contiguous_domains", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()

    parse_ok = (
        _to_bool_series(domains_df["parse_ok"])
        if "parse_ok" in domains_df.columns
        else pd.Series(False, index=domains_df.index)
    )
    domain_lengths = pd.to_numeric(domains_df.loc[parse_ok, "domain_length"], errors="coerce")
    domain_lengths = domain_lengths[(domain_lengths > 0) & domain_lengths.notna()]
    median_domain_length = domain_lengths.median() if not domain_lengths.empty else np.nan
    mean_domain_length = domain_lengths.mean() if not domain_lengths.empty else np.nan

    cath_assigned = (
        _to_bool_series(domains_df["cath_assigned"])
        if "cath_assigned" in domains_df.columns
        else pd.Series(False, index=domains_df.index)
    )
    assigned_ok = domains_df[parse_ok & cath_assigned].copy()
    assigned_ok = _build_cath_label_columns(assigned_ok) if not assigned_ok.empty else assigned_ok

    top_classes = (
        assigned_ok["cath_class_label"].dropna().astype(str).value_counts()
        if "cath_class_label" in assigned_ok.columns
        else pd.Series(dtype=int)
    )
    top_architectures = (
        assigned_ok["cath_architecture_label"].dropna().astype(str).value_counts()
        if "cath_architecture_label" in assigned_ok.columns
        else pd.Series(dtype=int)
    )
    top_topologies = (
        assigned_ok["cath_topology_label"].dropna().astype(str).value_counts()
        if "cath_topology_label" in assigned_ok.columns
        else pd.Series(dtype=int)
    )

    rows: List[Dict[str, str]] = [
        {"Metric": "Valid chains filtered", "Value": _format_count(valid_chain_count)},
        {
            "Metric": "Excluded missing chains present in chains file",
            "Value": _format_count(int(eda_summary.get("excluded_by_missing_count", 0))),
        },
        {"Metric": "Total parsed domains", "Value": _format_count(parsed_domains_count)},
        {"Metric": "Domain parse failures", "Value": _format_count(parse_failures)},
        {"Metric": "Domain parse success rate", "Value": _format_percent(parse_success_rate, decimals=2)},
        {"Metric": "Assigned domains", "Value": _format_count(assigned_domains_count)},
        {"Metric": "Unassigned domains", "Value": _format_count(unassigned_domains_count)},
        {"Metric": "Assigned fraction", "Value": _format_percent(assigned_fraction, decimals=2)},
        {"Metric": "Mean domains per chain", "Value": _format_float(mean_domains_per_chain, decimals=2)},
        {"Metric": "Median domains per chain", "Value": _format_float(median_domains_per_chain, decimals=2)},
        {
            "Metric": "Fraction of chains with any non-contiguous domain",
            "Value": _format_percent(frac_chains_with_non_contig, decimals=2),
        },
        {"Metric": "Total non-contiguous domains", "Value": _format_count(total_non_contig_domains)},
        {"Metric": "Median domain length aa", "Value": _format_float(median_domain_length, decimals=2)},
        {"Metric": "Mean domain length aa", "Value": _format_float(mean_domain_length, decimals=2)},
    ]

    for idx in range(3):
        if idx < len(top_classes):
            label = str(top_classes.index[idx])
            count = int(top_classes.iloc[idx])
            value = f"{label}:{count:,}"
        else:
            value = "N/A"
        rows.append({"Metric": f"Top CATH class {idx + 1}", "Value": value})

    for idx in range(5):
        if idx < len(top_architectures):
            label = str(top_architectures.index[idx])
            count = int(top_architectures.iloc[idx])
            value = f"{label}:{count:,}"
        else:
            value = "N/A"
        rows.append({"Metric": f"Top CATH architecture {idx + 1}", "Value": value})

    for idx in range(3):
        if idx < len(top_topologies):
            label = str(top_topologies.index[idx])
            count = int(top_topologies.iloc[idx])
            value = f"{label}:{count:,}"
        else:
            value = "N/A"
        rows.append({"Metric": f"Top CATH topology {idx + 1}", "Value": value})

    rows.append(
        {
            "Metric": "Training chains count",
            "Value": _detect_training_chains_count(filtered_df),
        }
    )
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _write_poster_stats_markdown(stats_df: pd.DataFrame, md_path: Path) -> None:
    """Write poster stats table as Markdown."""

    def esc(text: object) -> str:
        return str(text).replace("|", "\\|")

    lines = ["| Metric | Value |", "|---|---|"]
    for _, row in stats_df.iterrows():
        lines.append(f"| {esc(row['Metric'])} | {esc(row['Value'])} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_poster_stats_table_figure(stats_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Render poster stats as a compact matplotlib table figure."""
    n_rows = len(stats_df)
    fig_height = min(14.0, max(5.0, 1.2 + 0.34 * n_rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=stats_df.values.tolist(),
        colLabels=stats_df.columns.tolist(),
        colLoc="left",
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.22)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#E9ECEF")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F8F9FA")
        cell.set_edgecolor("#D0D7DE")
        cell.set_linewidth(0.45)
    ax.set_title("Poster Stats Summary", fontsize=13, pad=8)
    save_figure(fig, "poster_stats_table", figures_dir, dpi)


def write_poster_stats_outputs(
    filtered_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    chain_summary_df: pd.DataFrame,
    eda_summary: Dict[str, object],
    cleaned_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> Tuple[Path, Path]:
    """Write compact poster stats CSV/Markdown and render optional table figure."""
    stats_df = _build_poster_stats_dataframe(
        filtered_df=filtered_df,
        domains_df=domains_df,
        chain_summary_df=chain_summary_df,
        eda_summary=eda_summary,
    )
    csv_path = cleaned_dir / "poster_stats_table.csv"
    md_path = cleaned_dir / "poster_stats_table.md"
    stats_df.to_csv(csv_path, index=False)
    _write_poster_stats_markdown(stats_df, md_path)
    _render_poster_stats_table_figure(stats_df, figures_dir, dpi)
    return csv_path, md_path


def plot_domain_length_by_contiguity(
    domains_df: pd.DataFrame, figures_dir: Path, dpi: int, max_violin_samples: int
) -> None:
    """Compare domain length distributions between contiguous and non-contiguous domains."""
    contig, non_contig = _get_contiguity_length_arrays(domains_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_domain_length_box_on_ax(
        ax=ax,
        contig=contig,
        non_contig=non_contig,
        title="Domain Length: Contiguous vs Non-contiguous Boxplot",
        positions=(1.0, 1.35),
    )
    save_figure(fig, "domain_length_by_contiguity_boxplot", figures_dir, dpi)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_domain_length_violin_on_ax(
        ax=ax,
        contig=contig,
        non_contig=non_contig,
        max_violin_samples=max_violin_samples,
        title="Domain Length: Contiguous vs Non-contiguous",
        positions=(1.0, 1.35),
    )
    save_figure(fig, "domain_length_by_contiguity_violin", figures_dir, dpi)


def plot_non_contiguous_domain_stats(
    chain_summary_df: pd.DataFrame, figures_dir: Path, dpi: int
) -> None:
    """Plot non-contiguous domain metrics per chain and global summary."""
    values = chain_summary_df["n_non_contiguous_domains"].to_numpy(dtype=float)
    if values.size == 0:
        fig = make_empty_figure(
            "Non-contiguous Domains per Chain",
            "No chain summary data available.",
        )
        save_figure(fig, "non_contiguous_domains_per_chain_hist", figures_dir, dpi)
        fig = make_empty_figure(
            "Contiguous vs Non-contiguous Domains",
            "No chain summary data available.",
        )
        save_figure(fig, "non_contiguous_domain_summary_bar", figures_dir, dpi)
        return

    max_val = int(np.nanmax(values))
    bins = np.arange(-0.5, max_val + 1.5, 1.0) if max_val <= 120 else _auto_hist_bins(values)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins, color="#577590", alpha=0.9, edgecolor="black", linewidth=0.3)
    ax.set_title("Distribution of Non-contiguous Domains per Chain")
    ax.set_xlabel("Non-contiguous Domains per Chain")
    ax.set_ylabel("Number of Chains")
    save_figure(fig, "non_contiguous_domains_per_chain_hist", figures_dir, dpi)

    total_domains = int(chain_summary_df["n_domains"].sum())
    total_non = int(chain_summary_df["n_non_contiguous_domains"].sum())
    total_contig = max(0, total_domains - total_non)
    frac_non = (total_non / total_domains) if total_domains else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["Contiguous", "Non-contiguous"],
        [total_contig, total_non],
        color=["#43AA8B", "#F94144"],
    )
    ax.set_title("Contiguous vs Non-contiguous Domain Counts")
    ax.set_ylabel("Number of Domains")
    ax.text(
        0.5,
        max(total_contig, total_non) * 0.95 if max(total_contig, total_non) > 0 else 0.0,
        f"Fraction non-contiguous: {frac_non:.3%}",
        ha="center",
        va="top",
        fontsize=11,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
        )
    save_figure(fig, "non_contiguous_domain_summary_bar", figures_dir, dpi)


def plot_chain_count_summary(
    original_count: int, filtered_count: int, excluded_count: int, figures_dir: Path, dpi: int
) -> None:
    """Plot summary counts for original/filtered/excluded chain rows."""
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Original chains", "Filtered valid chains", "Excluded missing chains"]
    values = [original_count, filtered_count, excluded_count]
    bars = ax.bar(labels, values, color=["#4D908E", "#277DA1", "#F3722C"])
    ax.set_title("Chain Row Counts Summary")
    ax.set_ylabel("Number of Chain Rows")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
        )
    save_figure(fig, "chain_counts_summary_bar", figures_dir, dpi)


def _top_bar_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    figures_dir: Path,
    dpi: int,
    top_n: int = 15,
    color: str = "#277DA1",
) -> None:
    """Render a horizontal bar chart for top categories."""
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_top_barh_on_ax(
        ax=ax,
        series=series,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        top_n=top_n,
        color=color,
    )
    save_figure(fig, stem, figures_dir, dpi)


def plot_eda_poster_panel(
    chain_summary_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    max_violin_samples: int,
) -> None:
    """Create a composite poster panel of key EDA views in one multi-panel figure."""
    assigned = _prep_assigned_domains(domains_df)
    assigned = _build_cath_label_columns(assigned)
    contig, non_contig = _get_contiguity_length_arrays(domains_df)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    _draw_domain_length_violin_on_ax(
        ax=axes[0, 0],
        contig=contig,
        non_contig=non_contig,
        max_violin_samples=max_violin_samples,
        title="A. Domain Length: Contiguous vs Non-contiguous",
        positions=(1.0, 1.35),
    )
    _draw_top_barh_on_ax(
        ax=axes[0, 1],
        series=assigned["cath_class_label"] if not assigned.empty else pd.Series(dtype="string"),
        title="B. Top CATH Classes",
        xlabel="Number of Domains",
        ylabel="CATH Class",
        top_n=10,
        color="#264653",
    )
    _draw_top_barh_on_ax(
        ax=axes[1, 0],
        series=assigned["cath_architecture_label"] if not assigned.empty else pd.Series(dtype="string"),
        title="C. Top CATH Architectures",
        xlabel="Number of Domains",
        ylabel="CATH Architecture",
        top_n=12,
        color="#457B9D",
    )
    _draw_domains_per_chain_hist_on_ax(
        ax=axes[1, 1],
        chain_summary_df=chain_summary_df,
        title="D. Domains per Chain",
        color="#1B998B",
        zoom_percentile=99.5,
    )
    fig.suptitle("TED EDA Poster Panel", fontsize=16)
    save_figure(fig, "eda_poster_panel", figures_dir, dpi)


def plot_eda_poster_panel_1x4(
    filtered_df: pd.DataFrame,
    chain_summary_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    max_violin_samples: int,
) -> None:
    """Create a 1x4 poster-style panel combining key TED EDA visuals."""
    assigned = _prep_assigned_domains(domains_df)
    assigned = _build_cath_label_columns(assigned)
    contig, non_contig = _get_contiguity_length_arrays(domains_df)
    domain_lengths, seq_lengths, _ = _prepare_domain_vs_sequence_lengths(filtered_df, domains_df)

    fig, axes = plt.subplots(1, 4, figsize=(27, 6.0), constrained_layout=True)
    _draw_domain_length_violin_on_ax(
        ax=axes[0],
        contig=contig,
        non_contig=non_contig,
        max_violin_samples=max_violin_samples,
        title="A. Domain Length: Contiguous vs Non-contiguous",
        positions=(1.0, 1.35),
    )
    _draw_domain_vs_sequence_overlay_on_ax(
        ax=axes[1],
        domain_lengths=domain_lengths,
        sequence_lengths=seq_lengths,
        title="B. Domain vs Sequence Length",
    )
    _draw_top_barh_on_ax(
        ax=axes[2],
        series=assigned["cath_architecture_label"] if not assigned.empty else pd.Series(dtype="string"),
        title="C. Top CATH Architectures",
        xlabel="Number of Domains",
        ylabel="CATH Architecture",
        top_n=12,
        color="#457B9D",
        gradient=True,
        colormap="cividis",
    )
    _draw_domains_per_chain_hist_on_ax(
        ax=axes[3],
        chain_summary_df=chain_summary_df,
        title="D. Domains per Chain",
        color="#1B998B",
        zoom_percentile=99.5,
    )
    fig.suptitle("Exploratory Data Analysis of TED Domain Annotations", fontsize=15)
    save_figure(fig, "eda_poster_panel_1x4", figures_dir, dpi)


def build_pruned_cath_tree(
    assigned_df: pd.DataFrame, prune_cfg: TreePruneConfig
) -> Tuple[List[Dict[str, object]], List[Tuple[str, str, int]]]:
    """
    Build a pruned CATH hierarchy for deterministic static tree plotting.

    Node dict keys: node_id, parent_id, level, label, count.
    Edge tuple: parent_id, child_id, count.
    """
    nodes: List[Dict[str, object]] = []
    edges: List[Tuple[str, str, int]] = []

    total_count = int(len(assigned_df))
    nodes.append(
        {
            "node_id": "root",
            "parent_id": None,
            "level": 0,
            "label": "root",
            "count": total_count,
        }
    )
    if total_count == 0:
        return nodes, edges

    class_counts = assigned_df["cath_class_label"].dropna().value_counts()
    top_classes = class_counts.head(prune_cfg.top_classes)

    for class_label, class_count in top_classes.items():
        class_id = f"class:{class_label}"
        nodes.append(
            {
                "node_id": class_id,
                "parent_id": "root",
                "level": 1,
                "label": str(class_label),
                "count": int(class_count),
            }
        )
        edges.append(("root", class_id, int(class_count)))

        class_df = assigned_df[assigned_df["cath_class_label"] == class_label]
        arch_counts = class_df["cath_architecture_label"].dropna().value_counts()
        top_arch = arch_counts.head(prune_cfg.top_arch_per_class)

        for arch_label, arch_count in top_arch.items():
            arch_id = f"arch:{arch_label}"
            nodes.append(
                {
                    "node_id": arch_id,
                    "parent_id": class_id,
                    "level": 2,
                    "label": str(arch_label),
                    "count": int(arch_count),
                }
            )
            edges.append((class_id, arch_id, int(arch_count)))

            arch_df = class_df[class_df["cath_architecture_label"] == arch_label]
            topo_counts = arch_df["cath_topology_label"].dropna().value_counts()
            top_topo = topo_counts.head(prune_cfg.top_topo_per_arch)

            for topo_label, topo_count in top_topo.items():
                topo_id = f"topo:{topo_label}"
                nodes.append(
                    {
                        "node_id": topo_id,
                        "parent_id": arch_id,
                        "level": 3,
                        "label": str(topo_label),
                        "count": int(topo_count),
                    }
                )
                edges.append((arch_id, topo_id, int(topo_count)))

                topo_df = arch_df[arch_df["cath_topology_label"] == topo_label]
                sf_counts = topo_df["cath_superfamily_label"].dropna().value_counts()
                top_sf = sf_counts.head(prune_cfg.top_superfamily_per_topo)

                for sf_label, sf_count in top_sf.items():
                    sf_id = f"sf:{sf_label}"
                    nodes.append(
                        {
                            "node_id": sf_id,
                            "parent_id": topo_id,
                            "level": 4,
                            "label": str(sf_label),
                            "count": int(sf_count),
                        }
                    )
                    edges.append((topo_id, sf_id, int(sf_count)))

    return nodes, edges


def _theta_arc(theta_start: float, theta_end: float, n_points: int = 28) -> np.ndarray:
    """Return theta samples along the shortest angular arc from start to end."""
    delta = (theta_end - theta_start + np.pi) % (2.0 * np.pi) - np.pi
    return np.linspace(theta_start, theta_start + delta, n_points)


def _prepare_circular_hierarchy(
    assigned_df: pd.DataFrame,
    circular_cfg: CircularPlotConfig,
) -> Tuple[Dict[str, Dict[str, object]], List[Tuple[str, str, int]], Dict[str, List[str]], List[str], int]:
    """
    Prepare pruned hierarchy structures for circular plotting.

    Returns:
      node_map, filtered_edges, children_map, ordered_leaf_ids, actual_leaf_level
    """
    leaf_level = 3 if circular_cfg.leaf_level == "topology" else 4
    required_cols = ["cath_class_label", "cath_architecture_label", "cath_topology_label"]
    if leaf_level == 4:
        required_cols.append("cath_superfamily_label")

    hierarchy_df = assigned_df[required_cols].dropna().copy()
    if hierarchy_df.empty:
        return {}, [], {}, [], leaf_level

    leaf_col = "cath_topology_label" if leaf_level == 3 else "cath_superfamily_label"
    leaf_counts = (
        hierarchy_df.groupby(leaf_col, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(by=["count", leaf_col], ascending=[False, True], kind="mergesort")
    )
    max_leaves = max(1, int(circular_cfg.max_leaves))
    selected_leaf_labels = leaf_counts.head(max_leaves)[leaf_col].astype(str).tolist()
    hierarchy_df = hierarchy_df[hierarchy_df[leaf_col].astype(str).isin(selected_leaf_labels)].copy()
    if hierarchy_df.empty:
        return {}, [], {}, [], leaf_level

    class_counts = hierarchy_df.groupby("cath_class_label").size().to_dict()
    arch_counts = hierarchy_df.groupby("cath_architecture_label").size().to_dict()
    topo_counts = hierarchy_df.groupby("cath_topology_label").size().to_dict()
    sf_counts = (
        hierarchy_df.groupby("cath_superfamily_label").size().to_dict()
        if leaf_level == 4
        else {}
    )

    node_map: Dict[str, Dict[str, object]] = {
        "root": {"node_id": "root", "parent_id": None, "level": 0, "label": "root", "count": int(len(hierarchy_df))}
    }

    for class_label, count in class_counts.items():
        node_id = f"class:{class_label}"
        node_map[node_id] = {
            "node_id": node_id,
            "parent_id": "root",
            "level": 1,
            "label": str(class_label),
            "count": int(count),
        }

    arch_parent = (
        hierarchy_df[["cath_architecture_label", "cath_class_label"]]
        .drop_duplicates()
        .set_index("cath_architecture_label")["cath_class_label"]
        .to_dict()
    )
    for arch_label, count in arch_counts.items():
        class_label = arch_parent.get(arch_label)
        if class_label is None:
            continue
        parent_id = f"class:{class_label}"
        if parent_id not in node_map:
            continue
        node_id = f"arch:{arch_label}"
        node_map[node_id] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "level": 2,
            "label": str(arch_label),
            "count": int(count),
        }

    topo_parent = (
        hierarchy_df[["cath_topology_label", "cath_architecture_label"]]
        .drop_duplicates()
        .set_index("cath_topology_label")["cath_architecture_label"]
        .to_dict()
    )
    for topo_label, count in topo_counts.items():
        arch_label = topo_parent.get(topo_label)
        if arch_label is None:
            continue
        parent_id = f"arch:{arch_label}"
        if parent_id not in node_map:
            continue
        node_id = f"topo:{topo_label}"
        node_map[node_id] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "level": 3,
            "label": str(topo_label),
            "count": int(count),
        }

    if leaf_level == 4:
        sf_parent = (
            hierarchy_df[["cath_superfamily_label", "cath_topology_label"]]
            .drop_duplicates()
            .set_index("cath_superfamily_label")["cath_topology_label"]
            .to_dict()
        )
        for sf_label, count in sf_counts.items():
            topo_label = sf_parent.get(sf_label)
            if topo_label is None:
                continue
            parent_id = f"topo:{topo_label}"
            if parent_id not in node_map:
                continue
            node_id = f"sf:{sf_label}"
            node_map[node_id] = {
                "node_id": node_id,
                "parent_id": parent_id,
                "level": 4,
                "label": str(sf_label),
                "count": int(count),
            }

    filtered_edges: List[Tuple[str, str, int]] = []
    for node_id, node in node_map.items():
        parent_id = node["parent_id"]
        if parent_id is None:
            continue
        filtered_edges.append((str(parent_id), str(node_id), int(node["count"])))

    children_map: Dict[str, List[str]] = {node_id: [] for node_id in node_map}
    for parent_id, child_id, _ in filtered_edges:
        if parent_id in children_map:
            children_map[parent_id].append(child_id)
    for parent_id, child_ids in children_map.items():
        child_ids.sort(
            key=lambda node_id: (
                str(node_map[node_id]["label"]),
                -int(node_map[node_id]["count"]),
            )
        )

    leaf_prefix = "topo:" if leaf_level == 3 else "sf:"
    selected_leaf_ids = [f"{leaf_prefix}{label}" for label in selected_leaf_labels if f"{leaf_prefix}{label}" in node_map]
    return node_map, filtered_edges, children_map, selected_leaf_ids, leaf_level


def _order_leaves_for_circle(
    leaf_ids: List[str],
    node_map: Dict[str, Dict[str, object]],
) -> List[str]:
    """Order leaves deterministically using lineage labels for grouped circular layout."""

    def lineage(node_id: str) -> Tuple[str, ...]:
        labels: List[str] = []
        current = node_id
        while current is not None and current in node_map:
            labels.append(str(node_map[current]["label"]))
            current = node_map[current]["parent_id"]
        return tuple(reversed(labels))

    return sorted(
        leaf_ids,
        key=lambda node_id: (lineage(node_id), -int(node_map[node_id]["count"]), str(node_map[node_id]["label"])),
    )


def _compute_circular_angles(
    node_map: Dict[str, Dict[str, object]],
    children_map: Dict[str, List[str]],
    ordered_leaf_ids: List[str],
) -> Dict[str, float]:
    """Compute deterministic node angles by placing leaves evenly and averaging upward."""
    if not ordered_leaf_ids:
        return {}

    thetas = np.linspace(0.0, 2.0 * np.pi, len(ordered_leaf_ids), endpoint=False)
    angle_map = {leaf_id: float(theta) for leaf_id, theta in zip(ordered_leaf_ids, thetas)}

    def compute_angle(node_id: str) -> float:
        if node_id in angle_map:
            return angle_map[node_id]
        children = children_map.get(node_id, [])
        if not children:
            angle_map[node_id] = 0.0
            return 0.0
        child_angles = np.array([compute_angle(child) for child in children], dtype=float)
        mean_complex = np.exp(1j * child_angles).mean()
        angle = float(np.angle(mean_complex))
        if angle < 0.0:
            angle += 2.0 * np.pi
        angle_map[node_id] = angle
        return angle

    for node_id in node_map:
        compute_angle(node_id)
    return angle_map


def _select_circular_leaf_labels(
    ordered_leaf_ids: List[str],
    leaf_thetas: np.ndarray,
    node_map: Dict[str, Dict[str, object]],
    target_labels: int,
    max_candidates: int,
    min_label_angle_deg: float,
    minimum_final_labels: int = 20,
    minimum_allowed_angle_deg: float = 4.0,
) -> List[str]:
    """
    Select readable leaf labels with angular collision filtering.

    Strategy:
    - Start from top-K leaves by count (importance-first candidates).
    - Sort candidates by angle and greedily enforce a minimum angular gap.
    - If too few labels survive, relax angular gap down to a floor.
    """
    if not ordered_leaf_ids:
        return []

    leaf_index = {node_id: idx for idx, node_id in enumerate(ordered_leaf_ids)}
    ranked_by_count = sorted(
        ordered_leaf_ids,
        key=lambda node_id: (-int(node_map[node_id]["count"]), str(node_map[node_id]["label"])),
    )
    candidate_n = max(1, min(int(max_candidates), len(ranked_by_count)))
    candidate_ids = ranked_by_count[:candidate_n]

    candidate_items = []
    for node_id in candidate_ids:
        idx = leaf_index[node_id]
        angle_deg = float(np.degrees(leaf_thetas[idx]) % 360.0)
        count = int(node_map[node_id]["count"])
        candidate_items.append((node_id, angle_deg, count))
    candidate_items.sort(key=lambda item: item[1])

    target = max(1, int(target_labels))
    min_needed = min(len(candidate_items), target, max(1, int(minimum_final_labels)))

    def place_with_gap(min_gap_deg: float) -> List[Tuple[str, float, int]]:
        selected: List[Tuple[str, float, int]] = []
        last_angle: Optional[float] = None
        for item in candidate_items:
            _, angle_deg, _ = item
            if last_angle is None or (angle_deg - last_angle) >= min_gap_deg:
                selected.append(item)
                last_angle = angle_deg
                if len(selected) >= target:
                    break

        while len(selected) > 1:
            first = selected[0]
            last = selected[-1]
            wrap_gap = (first[1] + 360.0) - last[1]
            if wrap_gap >= min_gap_deg:
                break
            if first[2] <= last[2]:
                selected.pop(0)
            else:
                selected.pop()
        return selected

    gap = max(float(min_label_angle_deg), float(minimum_allowed_angle_deg))
    selected_items = place_with_gap(gap)
    while len(selected_items) < min_needed and gap > minimum_allowed_angle_deg:
        gap = max(minimum_allowed_angle_deg, gap - 1.0)
        selected_items = place_with_gap(gap)
        if gap <= minimum_allowed_angle_deg:
            break

    return [item[0] for item in selected_items]


def _plot_cath_hierarchy_circular(
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    prune_cfg: TreePruneConfig,
    circular_cfg: CircularPlotConfig,
    output_stem: str,
) -> None:
    """Plot a circular, circos-like overview of the pruned CATH hierarchy."""
    _ = prune_cfg  # kept for API compatibility with existing pipeline wiring
    assigned = _prep_assigned_domains(domains_df)
    assigned = _build_cath_label_columns(assigned)
    requested_leaf_level = "superfamily" if circular_cfg.leaf_level == "superfamily" else "topology"
    requested_leaf_name = "Superfamily" if requested_leaf_level == "superfamily" else "Topology"

    if assigned.empty:
        fig = make_empty_figure(
            f"CATH Circular Hierarchy Overview {requested_leaf_name}",
            "No assigned CATH labels available.",
        )
        save_figure(fig, output_stem, figures_dir, dpi)
        return

    (
        node_map,
        filtered_edges,
        children_map,
        selected_leaf_ids,
        leaf_level,
    ) = _prepare_circular_hierarchy(assigned, circular_cfg)
    if not selected_leaf_ids:
        fig = make_empty_figure(
            f"CATH Circular Hierarchy Overview {requested_leaf_name}",
            "No pruned hierarchy leaves available.",
        )
        save_figure(fig, output_stem, figures_dir, dpi)
        return

    ordered_leaf_ids = _order_leaves_for_circle(selected_leaf_ids, node_map)
    angle_map = _compute_circular_angles(node_map, children_map, ordered_leaf_ids)

    if leaf_level == 4:
        level_radius = {0: 0.45, 1: 1.30, 2: 2.10, 3: 2.90, 4: 3.70}
    else:
        level_radius = {0: 0.45, 1: 1.40, 2: 2.25, 3: 3.10}

    fig = plt.figure(figsize=(11.0, 11.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.972, bottom=0.02)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    max_edge_count = max((count for _, _, count in filtered_edges), default=1)
    max_node_count = max((int(node["count"]) for node in node_map.values()), default=1)

    for parent, child, count in filtered_edges:
        theta_parent = angle_map[parent]
        theta_child = angle_map[child]
        r_parent = level_radius[int(node_map[parent]["level"])]
        r_child = level_radius[int(node_map[child]["level"])]
        child_level = int(node_map[child]["level"])
        ratio = float(count) / max_edge_count
        depth_alpha_scale = {1: 0.92, 2: 0.85, 3: 0.74, 4: 0.64}.get(child_level, 0.70)
        depth_width_scale = {1: 1.65, 2: 1.50, 3: 1.32, 4: 1.18}.get(child_level, 1.2)
        line_width = (0.58 + 2.75 * (ratio**0.75)) * depth_width_scale
        line_alpha = (0.12 + 0.55 * (ratio**0.70)) * depth_alpha_scale

        theta_arc = _theta_arc(theta_parent, theta_child, n_points=24)
        r_arc = np.full_like(theta_arc, r_parent)
        ax.plot(theta_arc, r_arc, color="#64748B", alpha=line_alpha, linewidth=line_width, zorder=1)
        ax.plot(
            [theta_child, theta_child],
            [r_parent, r_child],
            color="#64748B",
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1,
        )

    color_by_level = {
        0: "#1D3557",
        1: "#2A9D8F",
        2: "#4D908E",
        3: "#577590",
        4: "#6D597A",
    }
    for node_id, node in node_map.items():
        level = int(node["level"])
        theta = angle_map[node_id]
        radius = level_radius[level]
        count = int(node["count"])
        node_ratio = float(count) / max_node_count
        is_leaf = node_id in ordered_leaf_ids
        level_size_scale = {0: 1.45, 1: 1.25, 2: 1.10, 3: 0.95, 4: 0.85}.get(level, 1.0)
        size = (14.0 + 85.0 * node_ratio) * level_size_scale
        alpha = 0.68 if is_leaf else (0.90 if level <= 2 else 0.78)
        ax.scatter(
            [theta],
            [radius],
            s=size,
            color=color_by_level.get(level, "#4D908E"),
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    leaf_counts = np.array([int(node_map[node_id]["count"]) for node_id in ordered_leaf_ids], dtype=float)
    max_leaf_count = float(max(leaf_counts.max(), 1.0))
    leaf_thetas = np.array([angle_map[node_id] for node_id in ordered_leaf_ids], dtype=float)
    bar_width = (2.0 * np.pi / len(ordered_leaf_ids)) * 0.82
    outer_base = max(level_radius.values()) + 0.45
    outer_band = 1.25
    bar_heights = outer_band * (leaf_counts / max_leaf_count)
    ax.bar(
        leaf_thetas,
        bar_heights,
        width=bar_width,
        bottom=outer_base,
        color="#F4A261",
        edgecolor="none",
        alpha=0.72,
        align="center",
        zorder=2,
    )

    default_target = 40 if leaf_level == 3 else 50
    effective_label_target = max(default_target, int(circular_cfg.label_target), int(circular_cfg.leaf_label_top_n))
    default_candidate_cap = 40 if leaf_level == 3 else 60
    effective_candidate_cap = max(default_candidate_cap, int(circular_cfg.label_max_candidates))
    label_ids = _select_circular_leaf_labels(
        ordered_leaf_ids=ordered_leaf_ids,
        leaf_thetas=leaf_thetas,
        node_map=node_map,
        target_labels=effective_label_target,
        max_candidates=effective_candidate_cap,
        min_label_angle_deg=float(circular_cfg.min_label_angle_deg),
        minimum_final_labels=20,
        minimum_allowed_angle_deg=4.0,
    )
    label_set = set(label_ids)
    label_padding = 0.22
    tier_offset = 0.15
    label_counter = 0
    for idx, node_id in enumerate(ordered_leaf_ids):
        if node_id not in label_set:
            continue
        theta = leaf_thetas[idx]
        label = str(node_map[node_id]["label"])
        deg = float(np.degrees(theta) % 360.0)
        if 90.0 < deg < 270.0:
            rotation = deg + 180.0
            horizontal_align = "right"
        else:
            rotation = deg
            horizontal_align = "left"

        bar_outer = outer_base + bar_heights[idx]
        label_radius = bar_outer + label_padding + (tier_offset if (label_counter % 2 == 1) else 0.0)

        # Subtle leader line from bar tip to label ring for cleaner association.
        ax.plot(
            [theta, theta],
            [bar_outer + 0.01, label_radius - 0.03],
            color="#475569",
            alpha=0.35,
            linewidth=0.6,
            zorder=2.5,
        )
        ax.text(
            theta,
            label_radius,
            label,
            fontsize=6.8,
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_align,
            va="center",
            color="#2F2F2F",
        )
        label_counter += 1

    leaf_name = "topologies" if leaf_level == 3 else "superfamilies"
    n_top_shown = len(ordered_leaf_ids)
    if leaf_level == 4:
        title_text = "CATH Circular Hierarchy Overview — Top 60 superfamilies"
    else:
        title_text = f"CATH Circular Hierarchy Overview — Top {n_top_shown} {leaf_name}"
    ax.set_title(title_text, pad=6)
    ax.set_ylim(0.0, outer_base + outer_band + 1.05)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#64748B", linewidth=2.0, label="Hierarchy links"),
            Patch(facecolor="#F4A261", alpha=0.72, label=f"{leaf_name.title()} count ring"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.03),
        frameon=False,
        fontsize=8.2,
    )

    save_figure(fig, output_stem, figures_dir, dpi)


def plot_cath_hierarchy_circular_overview(
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    prune_cfg: TreePruneConfig,
    circular_cfg: CircularPlotConfig,
) -> None:
    """Topology-leaf circular CATH hierarchy overview."""
    topo_cfg = CircularPlotConfig(
        max_leaves=max(1, int(circular_cfg.max_leaves)),
        leaf_label_top_n=max(0, int(circular_cfg.leaf_label_top_n)),
        leaf_level="topology",
        min_label_angle_deg=float(circular_cfg.min_label_angle_deg),
        label_target=max(0, int(circular_cfg.label_target)),
        label_max_candidates=max(1, int(circular_cfg.label_max_candidates)),
    )
    _plot_cath_hierarchy_circular(
        domains_df=domains_df,
        figures_dir=figures_dir,
        dpi=dpi,
        prune_cfg=prune_cfg,
        circular_cfg=topo_cfg,
        output_stem="cath_hierarchy_circular_overview",
    )


def plot_cath_hierarchy_circular_overview_superfamily(
    domains_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    prune_cfg: TreePruneConfig,
    circular_cfg: CircularPlotConfig,
) -> None:
    """Superfamily-leaf circular CATH hierarchy overview."""
    sf_cfg = CircularPlotConfig(
        max_leaves=max(1, int(circular_cfg.max_leaves)),
        leaf_label_top_n=max(0, int(circular_cfg.leaf_label_top_n)),
        leaf_level="superfamily",
        min_label_angle_deg=float(circular_cfg.min_label_angle_deg),
        label_target=max(0, int(circular_cfg.label_target)),
        label_max_candidates=max(1, int(circular_cfg.label_max_candidates)),
    )
    _plot_cath_hierarchy_circular(
        domains_df=domains_df,
        figures_dir=figures_dir,
        dpi=dpi,
        prune_cfg=prune_cfg,
        circular_cfg=sf_cfg,
        output_stem="cath_hierarchy_circular_overview_superfamily",
    )


def plot_cath_tree_pruned(
    assigned_df: pd.DataFrame, figures_dir: Path, dpi: int, prune_cfg: TreePruneConfig
) -> None:
    """
    Plot a deterministic, pruned layered CATH hierarchy.

    This is a matplotlib-native fallback to keep the hierarchy readable without
    requiring complex graph-layout dependencies.
    """
    nodes, edges = build_pruned_cath_tree(assigned_df, prune_cfg)
    if len(nodes) <= 1:
        fig = make_empty_figure("Pruned CATH Hierarchy", "No assigned CATH labels available.")
        save_figure(fig, "cath_hierarchy_tree_pruned", figures_dir, dpi)
        return

    # This figure is intentionally a pruned overview of the CATH hierarchy.
    # Fine-grained label detail is handled by the dedicated level bar plots.
    node_df = pd.DataFrame(nodes)
    edge_df = pd.DataFrame(edges, columns=["parent_id", "child_id", "count"])
    node_map = {row["node_id"]: row for row in nodes}

    level_order: Dict[int, List[str]] = {}
    prev_level_nodes: List[str] = ["root"]
    level_order[0] = prev_level_nodes
    for level in [1, 2, 3, 4]:
        level_nodes = node_df[node_df["level"] == level].copy()
        if level_nodes.empty:
            break
        parent_rank = {pid: i for i, pid in enumerate(prev_level_nodes)}
        level_nodes["parent_rank"] = level_nodes["parent_id"].map(parent_rank).fillna(10**9)
        level_nodes = level_nodes.sort_values(
            by=["parent_rank", "count", "label"], ascending=[True, False, True]
        )
        ordered_ids = level_nodes["node_id"].tolist()
        level_order[level] = ordered_ids
        prev_level_nodes = ordered_ids

    max_level = max(level_order.keys())

    # Readability controls (kept local for deterministic static rendering).
    LEVEL_GAP = 3.2
    TOPOLOGY_LABEL_MIN_COUNT = 1000
    label_superfamilies = False
    SUPERFAMILY_LABEL_MIN_COUNT = 5000
    MAX_LABELED_TOPOLOGY = 18
    MAX_LABELED_SUPERFAMILY = 8
    FONT_BY_LEVEL = {0: 11, 1: 10, 2: 9, 3: 7.5, 4: 6.5}
    LABEL_X_OFFSET_BY_LEVEL = {0: 0.20, 1: 0.24, 2: 0.30, 3: 0.40, 4: 0.50}
    x_positions = {lvl: float(lvl * LEVEL_GAP) for lvl in range(max_level + 1)}
    y_positions: Dict[str, float] = {}

    for level, node_ids in level_order.items():
        n = len(node_ids)
        if n == 1:
            y_vals = [0.5]
        else:
            y_vals = np.linspace(0.98, 0.02, n)
        for node_id, y in zip(node_ids, y_vals):
            y_positions[node_id] = float(y)

    # Decide which nodes get text labels:
    # - always root/class/architecture
    # - topology only for high-frequency nodes
    # - superfamily labels are hidden by default to keep this as an overview.
    labels_to_show: set[str] = set()
    for level in [0, 1, 2]:
        for node_id in level_order.get(level, []):
            labels_to_show.add(node_id)

    topo_nodes = [
        node for node in nodes if int(node["level"]) == 3 and int(node["count"]) >= TOPOLOGY_LABEL_MIN_COUNT
    ]
    topo_nodes = sorted(topo_nodes, key=lambda n: (-int(n["count"]), str(n["label"])))[:MAX_LABELED_TOPOLOGY]
    labels_to_show.update(str(node["node_id"]) for node in topo_nodes)

    sf_nodes: List[Dict[str, object]] = []
    if label_superfamilies:
        sf_nodes = [
            node
            for node in nodes
            if int(node["level"]) == 4 and int(node["count"]) >= SUPERFAMILY_LABEL_MIN_COUNT
        ]
        sf_nodes = sorted(sf_nodes, key=lambda n: (-int(n["count"]), str(n["label"])))[:MAX_LABELED_SUPERFAMILY]
        labels_to_show.update(str(node["node_id"]) for node in sf_nodes)

    fig, ax = plt.subplots(figsize=(24, 16))
    max_count = max(int(row["count"]) for row in nodes)
    max_edge = int(edge_df["count"].max()) if not edge_df.empty else 1

    for _, edge in edge_df.iterrows():
        parent = edge["parent_id"]
        child = edge["child_id"]
        x0, y0 = x_positions[node_map[parent]["level"]], y_positions[parent]
        x1, y1 = x_positions[node_map[child]["level"]], y_positions[child]
        edge_ratio = float(edge["count"]) / max_edge
        lw = 0.25 + 4.5 * (edge_ratio**0.80)
        edge_alpha = 0.10 + 0.55 * (edge_ratio**0.75)
        ax.plot([x0, x1], [y0, y1], color="#8D99AE", alpha=edge_alpha, linewidth=lw, zorder=1)

    for node in nodes:
        node_id = str(node["node_id"])
        level = int(node["level"])
        count = int(node["count"])
        label = str(node["label"])
        x = x_positions[level]
        y = y_positions[node_id]
        base_size = 70.0 + 850.0 * (count / max_count)
        is_labeled = node_id in labels_to_show
        if is_labeled:
            node_size = base_size * 1.08
            node_alpha = 0.92
            edge_color = "black"
            edge_width = 0.45
            zorder = 3
        else:
            node_size = base_size * (0.55 if level >= 3 else 0.72)
            node_alpha = 0.36 if level >= 3 else 0.50
            edge_color = "#404040"
            edge_width = 0.2
            zorder = 2

        ax.scatter(
            [x],
            [y],
            s=node_size,
            color="#2A9D8F",
            alpha=node_alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
            zorder=zorder,
        )

        if is_labeled:
            text = f"{label}\n(n={count:,})"
            x_offset = LABEL_X_OFFSET_BY_LEVEL.get(level, 0.24)
            fontsize = FONT_BY_LEVEL.get(level, 8)
            ax.text(
                x + x_offset,
                y,
                text,
                va="center",
                fontsize=fontsize,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.2},
            )

    max_x = x_positions[max_level]
    right_padding = 7.4
    ax.set_xlim(-0.6, max_x + right_padding)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([x_positions[level] for level in range(max_level + 1)])
    ax.set_xticklabels(["root", "class", "architecture", "topology", "superfamily"][: max_level + 1])
    ax.set_yticks([])
    ax.set_title("Pruned CATH Hierarchy (Assigned Domains)")
    ax.set_xlabel("Hierarchy Level")
    ax.grid(False)
    save_figure(fig, "cath_hierarchy_tree_pruned", figures_dir, dpi)


def plot_optional_extras(domains_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Generate optional extra plots requested by the user."""
    parse_ok = domains_df[domains_df["parse_ok"]].copy()

    seg_counts = pd.to_numeric(parse_ok["segments_count"], errors="coerce").dropna()
    if seg_counts.empty:
        fig = make_empty_figure("Segments per Domain", "No parseable domains available.")
    else:
        values = seg_counts.to_numpy(dtype=float)
        max_val = int(np.nanmax(values))
        bins = np.arange(0.5, max_val + 1.5, 1.0) if max_val <= 120 else _auto_hist_bins(values)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(values, bins=bins, color="#5F0F40", alpha=0.9, edgecolor="black", linewidth=0.3)
        ax.set_title("Distribution of Segments per Domain")
        ax.set_xlabel("Segments per Domain")
        ax.set_ylabel("Number of Domains")
    save_figure(fig, "segments_per_domain_hist", figures_dir, dpi)

    if parse_ok.empty:
        fig = make_empty_figure("Assigned vs Unassigned CATH", "No parseable domains available.")
        save_figure(fig, "cath_assigned_vs_unassigned_bar", figures_dir, dpi)
    else:
        assigned_count = int(parse_ok["cath_assigned"].fillna(False).sum())
        unassigned_count = int(len(parse_ok) - assigned_count)
        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.bar(
            ["Assigned", "Unassigned"],
            [assigned_count, unassigned_count],
            color=["#3A86FF", "#FF006E"],
        )
        ax.set_title("CATH Assigned vs Unassigned Domains")
        ax.set_ylabel("Number of Domains")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
            )
        save_figure(fig, "cath_assigned_vs_unassigned_bar", figures_dir, dpi)

    assigned = _prep_assigned_domains(domains_df)
    assigned = _build_cath_label_columns(assigned)
    _top_bar_plot(
        assigned["cath_topology_label"],
        title="Top CATH Topologies (Assigned Domains)",
        xlabel="Number of Domains",
        ylabel="CATH Topology",
        stem="top_cath_topology_bar",
        figures_dir=figures_dir,
        dpi=dpi,
        top_n=15,
        color="#4D908E",
    )


def plot_cath_summaries(
    domains_df: pd.DataFrame, figures_dir: Path, dpi: int, prune_cfg: TreePruneConfig
) -> None:
    """Generate required CATH class/architecture bars and pruned hierarchy figure."""
    assigned = _prep_assigned_domains(domains_df)
    assigned = _build_cath_label_columns(assigned)

    _top_bar_plot(
        assigned["cath_class_label"],
        title="Top CATH Classes",
        xlabel="Number of Domains",
        ylabel="CATH Class",
        stem="cath_class_top_bar",
        figures_dir=figures_dir,
        dpi=dpi,
        top_n=10,
        color="#264653",
    )

    _top_bar_plot(
        assigned["cath_architecture_label"],
        title="Top CATH Architectures",
        xlabel="Number of Domains",
        ylabel="CATH Architecture",
        stem="cath_architecture_top_bar",
        figures_dir=figures_dir,
        dpi=dpi,
        top_n=15,
        color="#457B9D",
    )

    plot_cath_tree_pruned(assigned, figures_dir, dpi, prune_cfg)


def write_outputs_readme(outputs_dir: Path) -> None:
    """Write a concise README describing generated outputs."""
    readme_path = outputs_dir / "README_outputs.txt"
    text = f"""TED EDA Outputs
================

Generated by: scripts/ted_eda.py (version {SCRIPT_VERSION})
Run command: python3 scripts/ted_eda.py

Folders
-------
- outputs/cleaned/: filtered and parsed data tables + QC summaries
- outputs/figures/: publication-style static figures (.png and .pdf)

Key cleaned files
-----------------
- chains.ted.filtered.csv: chain rows after removing missing accessions
- ted_domains.parsed.csv: one row per parsed domain token
- ted_chains.summary.csv: one row per unique uniprot_id with domain-level aggregates
- parsing_issues.csv: parse failures for malformed/empty tokens
- duplicate_uniprot_ids.csv: duplicate uniprot_id rows and counts
- eda_summary.json: global QC/EDA metrics and run metadata
- poster_stats_table.csv: compact poster-ready metric summary
- poster_stats_table.md: markdown version of compact poster-ready metrics
- domain_coverage_vector_1to2048.csv: mean domain coverage per position 1..2048
- domain_coverage_heatmap_matrix.csv: coverage matrix by sequence-length bins and positions
- domain_coverage_heatmap_bins.csv: sequence-length bin metadata for heatmap rows

Key figures
-----------
- domains_per_chain_hist(.png/.pdf) and zoomed variant
- domain_length_hist(.png/.pdf) and log-x variant
- domain_vs_sequence_length_overlay(.png/.pdf)
- domain_length_by_contiguity_boxplot(.png/.pdf)
- domain_length_by_contiguity_violin(.png/.pdf)
- non_contiguous_domains_per_chain_hist(.png/.pdf)
- non_contiguous_domain_summary_bar(.png/.pdf)
- chain_counts_summary_bar(.png/.pdf)
- cath_class_top_bar(.png/.pdf)
- cath_architecture_top_bar(.png/.pdf)
- cath_hierarchy_tree_pruned(.png/.pdf)
- eda_poster_panel(.png/.pdf)
- eda_poster_panel_1x4(.png/.pdf)
- cath_hierarchy_circular_overview(.png/.pdf)
- cath_hierarchy_circular_overview_superfamily(.png/.pdf)
- poster_stats_table(.png/.pdf)
- domain_coverage_distribution_1to2048(.png/.pdf)
- domain_coverage_heatmap_1to2048(.png/.pdf)

Optional extra figures
----------------------
- segments_per_domain_hist(.png/.pdf)
- cath_assigned_vs_unassigned_bar(.png/.pdf)
- top_cath_topology_bar(.png/.pdf)

Notes
-----
- Domain lengths are sums of inclusive segment lengths (end - start + 1).
- Non-contiguous domains are domain tokens with more than one segment split by '_'.
- Unassigned CATH labels are represented by '-'.
"""
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plot_style()

    ensure_dirs([args.outputs_dir, args.cleaned_dir, args.figures_dir])

    missing_accessions = load_missing_accessions([args.missing_a, args.missing_b])
    chains_df = load_chains_csv(args.chains_csv)
    original_rows = int(len(chains_df))
    print(f"[INFO] Original chain rows: {original_rows:,}")

    original_duplicates_df = find_duplicate_uniprot_ids(chains_df)
    if not original_duplicates_df.empty:
        print(
            f"[WARN] Duplicate uniprot_id rows detected in original CSV: "
            f"{len(original_duplicates_df):,} unique duplicate IDs."
        )

    excluded_mask = chains_df["uniprot_id"].isin(missing_accessions)
    excluded_present_count = int(excluded_mask.sum())
    filtered_df = chains_df.loc[~excluded_mask].copy()
    filtered_rows = int(len(filtered_df))

    assert filtered_rows == original_rows - excluded_present_count, (
        "Validation failed: filtered row count does not match original - excluded."
    )
    print(
        f"[FILTER] Excluded rows present in chains CSV: {excluded_present_count:,}; "
        f"remaining rows: {filtered_rows:,}"
    )

    filtered_duplicates_df = find_duplicate_uniprot_ids(filtered_df)
    dup_csv_path = args.cleaned_dir / "duplicate_uniprot_ids.csv"
    filtered_duplicates_df.to_csv(dup_csv_path, index=False)

    filtered_csv_path = args.cleaned_dir / "chains.ted.filtered.csv"
    filtered_df.to_csv(filtered_csv_path, index=False)
    print(f"[SAVE] {filtered_csv_path}")

    domains_df, parsing_issues_df = parse_domains(filtered_df)
    domains_csv_path = args.cleaned_dir / "ted_domains.parsed.csv"
    domains_df.to_csv(domains_csv_path, index=False)
    print(f"[SAVE] {domains_csv_path}")

    issues_csv_path = args.cleaned_dir / "parsing_issues.csv"
    parsing_issues_df.to_csv(issues_csv_path, index=False)
    print(f"[SAVE] {issues_csv_path}")

    chain_summary_df = build_chain_summary(filtered_df, domains_df)
    chain_summary_path = args.cleaned_dir / "ted_chains.summary.csv"
    chain_summary_df.to_csv(chain_summary_path, index=False)
    print(f"[SAVE] {chain_summary_path}")

    parsed_domain_count = int(len(domains_df))
    parse_fail_count = int((~domains_df["parse_ok"].fillna(False)).sum()) if not domains_df.empty else 0
    assigned_count = int(domains_df["cath_assigned"].fillna(False).sum()) if not domains_df.empty else 0
    unassigned_count = parsed_domain_count - assigned_count
    assigned_fraction = (assigned_count / parsed_domain_count) if parsed_domain_count else 0.0
    unassigned_fraction = (unassigned_count / parsed_domain_count) if parsed_domain_count else 0.0

    chains_with_non_contig = (
        int(chain_summary_df["has_non_contiguous_domain"].sum()) if not chain_summary_df.empty else 0
    )

    eda_summary = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_row_count": original_rows,
        "excluded_by_missing_count": excluded_present_count,
        "filtered_row_count": filtered_rows,
        "missing_accessions_total_unique_count": len(missing_accessions),
        "parsed_domains_count": parsed_domain_count,
        "domain_parse_failures_count": parse_fail_count,
        "assigned_domains_count": assigned_count,
        "unassigned_domains_count": unassigned_count,
        "assigned_fraction": assigned_fraction,
        "unassigned_fraction": unassigned_fraction,
        "chains_with_non_contiguous_domain_count": chains_with_non_contig,
        "duplicate_uniprot_ids": {
            "original_unique_duplicate_ids": int(len(original_duplicates_df)),
            "original_duplicate_rows_total": int(original_duplicates_df["row_count"].sum())
            if not original_duplicates_df.empty
            else 0,
            "filtered_unique_duplicate_ids": int(len(filtered_duplicates_df)),
            "filtered_duplicate_rows_total": int(filtered_duplicates_df["row_count"].sum())
            if not filtered_duplicates_df.empty
            else 0,
            "duplicate_rows_csv": str(dup_csv_path),
        },
    }

    summary_path = args.cleaned_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] {summary_path}")

    poster_stats_csv, poster_stats_md = write_poster_stats_outputs(
        filtered_df=filtered_df,
        domains_df=domains_df,
        chain_summary_df=chain_summary_df,
        eda_summary=eda_summary,
        cleaned_dir=args.cleaned_dir,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
    )
    print(f"[SAVE] {poster_stats_csv}")
    print(f"[SAVE] {poster_stats_md}")
    print("[INFO] Poster stats table written to outputs/cleaned and rendered to outputs/figures.")

    coverage_results = compute_domain_coverage_and_heatmap(
        filtered_df=filtered_df,
        domains_df=domains_df,
        cleaned_dir=args.cleaned_dir,
        sample_n=args.coverage_sample_n,
        max_pos=args.coverage_max_pos,
        bin_size=args.coverage_bin_size,
        rng_seed=RNG_SEED,
    )
    print(
        "[INFO] Domain coverage summary: "
        f"sample_size={coverage_results['sample_size']:,}, "
        f"max_pos={coverage_results['max_pos']}, "
        f"bins={coverage_results['n_bins']} "
        f"(bin_size={coverage_results['bin_size']})."
    )
    print(
        f"[INFO] Coverage vector source={coverage_results['seq_len_source']}; "
        "coverage files written to outputs/cleaned."
    )

    print("[PLOT] Generating figures...")
    plot_domains_per_chain(chain_summary_df, args.figures_dir, args.dpi)
    plot_domain_length_distributions(domains_df, args.figures_dir, args.dpi)
    plot_domain_vs_sequence_length_overlay(
        filtered_df=filtered_df,
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
    )
    plot_domain_coverage_distribution(
        coverage_vector_df=coverage_results["coverage_vector_df"],
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        max_pos=int(coverage_results["max_pos"]),
    )
    plot_domain_coverage_heatmap(
        heatmap_matrix=coverage_results["heatmap_matrix"],
        bin_labels=coverage_results["bin_labels"],
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        max_pos=int(coverage_results["max_pos"]),
        bin_size=int(coverage_results["bin_size"]),
    )
    plot_domain_length_by_contiguity(
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        max_violin_samples=args.max_violin_samples,
    )
    plot_non_contiguous_domain_stats(chain_summary_df, args.figures_dir, args.dpi)
    plot_chain_count_summary(
        original_count=original_rows,
        filtered_count=filtered_rows,
        excluded_count=excluded_present_count,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
    )

    prune_cfg = TreePruneConfig(
        top_classes=args.tree_top_classes,
        top_arch_per_class=args.tree_top_arch,
        top_topo_per_arch=args.tree_top_topo,
        top_superfamily_per_topo=args.tree_top_sf,
    )
    plot_cath_summaries(domains_df, args.figures_dir, args.dpi, prune_cfg)
    circular_cfg = CircularPlotConfig(
        max_leaves=max(1, int(args.circular_max_leaves)),
        leaf_label_top_n=max(0, int(args.circular_label_top)),
        leaf_level=args.circular_leaf_level,
        min_label_angle_deg=float(args.circular_min_label_angle_deg),
        label_target=max(0, int(args.circular_label_target)),
        label_max_candidates=max(1, int(args.circular_label_max_candidates)),
    )
    plot_cath_hierarchy_circular_overview(
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        prune_cfg=prune_cfg,
        circular_cfg=circular_cfg,
    )
    plot_cath_hierarchy_circular_overview_superfamily(
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        prune_cfg=prune_cfg,
        circular_cfg=circular_cfg,
    )
    plot_eda_poster_panel(
        chain_summary_df=chain_summary_df,
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        max_violin_samples=args.max_violin_samples,
    )
    plot_eda_poster_panel_1x4(
        filtered_df=filtered_df,
        chain_summary_df=chain_summary_df,
        domains_df=domains_df,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        max_violin_samples=args.max_violin_samples,
    )
    plot_optional_extras(domains_df, args.figures_dir, args.dpi)

    write_outputs_readme(args.outputs_dir)
    print(f"[SAVE] {args.outputs_dir / 'README_outputs.txt'}")

    print("[DONE] TED EDA pipeline finished.")
    print(
        "[DONE] Summary: "
        f"chains(original={original_rows:,}, excluded={excluded_present_count:,}, filtered={filtered_rows:,}), "
        f"domains(parsed={parsed_domain_count:,}, parse_fail={parse_fail_count:,}, "
        f"assigned={assigned_count:,}, unassigned={unassigned_count:,})."
    )


if __name__ == "__main__":
    main()
