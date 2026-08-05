#!/usr/bin/env python3
"""Compare CATH-1363 summary.json files across TED/Chainsaw runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = [
    "iou_chain",
    "correct_prop",
    "boundary_distance_score",
    "ndo",
    "domain_count_match",
    "correct_cath",
]
GROUPS = ["overall", "single", "multi"]


REFERENCE_ROWS: list[dict[str, Any]] = [
    {
        "label": "Gen1 greedy reference",
        "groups": {
            "overall": {
                "iou_chain": {"mean": 0.670},
                "boundary_distance_score": {"mean": 0.569},
            },
            "multi": {
                "iou_chain": {"mean": 0.475},
            },
        },
    },
    {
        "label": "Gen1 grammar reference",
        "groups": {
            "overall": {
                "iou_chain": {"mean": 0.675},
                "boundary_distance_score": {"mean": 0.562},
            },
            "multi": {
                "iou_chain": {"mean": 0.485},
            },
        },
    },
    {
        "label": "Chainsaw reference",
        "groups": {
            "overall": {
                "iou_chain": {"mean": 0.880},
                "boundary_distance_score": {"mean": 0.789},
            },
            "multi": {
                "iou_chain": {"mean": 0.832},
            },
        },
    },
]


def _norm_group(summary: dict[str, Any], group: str) -> dict[str, Any]:
    groups = summary.get("groups", {})
    if group in groups:
        return groups[group]
    aliases = {"overall": "1+ domains", "multi": "2+ domains"}
    alias = aliases.get(group)
    if alias and alias in groups:
        return groups[alias]
    return {}


def _metric_block(summary: dict[str, Any], group: str, metric: str) -> dict[str, float] | None:
    block = _norm_group(summary, group).get(metric)
    if isinstance(block, dict):
        return block
    return None


def _fmt(block: dict[str, float] | None) -> str:
    if not block:
        return "NA"
    mean = block.get("mean")
    if mean is None or not math.isfinite(float(mean)):
        return "NA"
    lo = block.get("ci_lo")
    hi = block.get("ci_hi")
    if lo is None or hi is None or not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def _ci(block: dict[str, float] | None) -> tuple[float, float] | None:
    if not block:
        return None
    lo = block.get("ci_lo")
    hi = block.get("ci_hi")
    if lo is None or hi is None:
        return None
    lo = float(lo)
    hi = float(hi)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    return lo, hi


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def load_rows(paths: list[Path], labels: list[str] | None) -> list[dict[str, Any]]:
    rows = []
    for i, path in enumerate(paths):
        summary = json.loads(path.read_text())
        label = labels[i] if labels and i < len(labels) else path.parent.name
        rows.append({"label": label, **summary})
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    all_rows = REFERENCE_ROWS + rows
    columns = ["run"] + [f"{group} {metric}" for group in GROUPS for metric in METRICS]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in all_rows:
        cells = [row["label"]]
        for group in GROUPS:
            for metric in METRICS:
                cells.append(_fmt(_metric_block(row, group, metric)))
        lines.append("| " + " | ".join(cells) + " |")

    warnings: list[str] = []
    for group in GROUPS:
        for metric in METRICS:
            with_ci = []
            for row in rows:
                ci = _ci(_metric_block(row, group, metric))
                if ci is not None:
                    with_ci.append((row["label"], ci))
            for i in range(len(with_ci)):
                for j in range(i + 1, len(with_ci)):
                    if _overlap(with_ci[i][1], with_ci[j][1]):
                        warnings.append(
                            f"- CI overlap: {group} {metric}: "
                            f"{with_ci[i][0]} overlaps {with_ci[j][0]}"
                        )
    if warnings:
        lines.extend(["", "## CI overlap warnings", *warnings])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path, help="Path(s) to results_*/summary.json")
    p.add_argument("--labels", nargs="*", default=None, help="Optional labels matching summaries.")
    p.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    args = p.parse_args()
    if args.labels is not None and len(args.labels) != len(args.summaries):
        p.error("--labels must have the same length as summaries")
    md = render_markdown(load_rows(args.summaries, args.labels))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
