#!/usr/bin/env python3
"""
Enrich CHAINSAW CATH1363 benchmark with CATH superfamily labels.

Duplicates data/processed/ into data/processed_with_cath/ and adds the real
CATH superfamily codes by querying the CATH REST API v4.3 for each domain.

The raw CHAINSAW CSV contains CATH domain names (e.g. '4w7sA_d1') in the
'true_dnames' column. These map directly to CATH domain IDs used to look up
superfamily codes.  The enriched CSV keeps the TED label format but replaces
the '-' CATH placeholders with actual codes (e.g. '3.40.50.300').

Usage:
    python add_cath_labels.py [--max_workers 4] [--retries 3] [--delay 0.3]
    python add_cath_labels.py --dry_run     # process first 10 rows only
"""

import argparse
import csv
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / "data" / "raw" / "chainsaw_model_v3_on_cath1363_test.csv"
INPUT_CSV = SCRIPT_DIR / "data" / "processed" / "chainsaw_cath1363_tedpred.csv"
OUT_DIR = SCRIPT_DIR / "data" / "processed_with_cath"
OUT_CSV = OUT_DIR / "chainsaw_cath1363_with_cath_labels.csv"
SUMMARY_JSON = OUT_DIR / "chainsaw_cath1363_cath_labels_summary.json"
CACHE_JSON = OUT_DIR / ".cath_label_cache.json"

CATH_API_BASE = "https://www.cathdb.info/version/v4_3_0/api/rest"

# TED label format: '<seg1>_<seg2> | <cath_label> * <seg3> | <cath_label>'
DOMAIN_SPLIT_RE = re.compile(r"\s*\*\s*")
# CHAINSAW domain name: '4w7sA_d1'
DNAME_RE = re.compile(r"^([0-9a-zA-Z]{4}[A-Za-z0-9])_d(\d+)$")


# ---------------------------------------------------------------------------
# CATH domain ID conversion
# ---------------------------------------------------------------------------

def dname_to_cath_ids(domain_name: str) -> list[str]:
    """
    Return candidate CATH API IDs for a CHAINSAW domain name.

    CHAINSAW:   '4w7sA_d1'  (1-based domain index)
    CATH API:   '4w7sA01'   (1-based, confirmed by live API)
                '4w7sA00'   (0-based, fallback just in case)
    """
    m = DNAME_RE.match(domain_name.strip())
    if not m:
        return []
    chain_part, num = m.group(1), int(m.group(2))
    # Try 1-based first (live API confirmed), then 0-based as fallback
    return [f"{chain_part}{num:02d}", f"{chain_part}{num - 1:02d}"]


# ---------------------------------------------------------------------------
# CATH REST API
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: float = 15.0) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TEDPred-benchmark/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _extract_cathcode(body: dict) -> str:
    """
    Extract the 4-level CATH superfamily code from the API response.

    The CATH v4.3 API returns:
      superfamily_id : '3.40.50.300'          ← what we want
      cath_id        : '3.40.50.300.395.2...' ← full FunFam ID (too granular)

    We prefer superfamily_id, then truncate any fallback to 4 levels.
    """
    data = body.get("data") or body
    # Preferred: 4-level superfamily code
    sfam = data.get("superfamily_id")
    if sfam and isinstance(sfam, str) and sfam not in ("-", ""):
        return sfam
    # Fallback: truncate cath_id / cathcode to 4 components
    for key in ("cath_id", "cathcode"):
        val = data.get(key)
        if val and isinstance(val, str) and val not in ("-", ""):
            parts = [p for p in val.split(".") if p]
            return ".".join(parts[:4]) if parts else "-"
    return "-"


def fetch_cath_superfamily(
    domain_name: str, retries: int = 3, delay: float = 0.3
) -> tuple[str, str, str]:
    """
    Query the CATH API for domain_name and return (cathcode, level, method).

    cathcode  : '3.40.50.300'  or '-'
    level     : 'H' (homologous superfamily), 'T' (topology/fold), or '-'
    method    : 'cath_database' or '-'
    """
    candidates = dname_to_cath_ids(domain_name)
    if not candidates:
        return "-", "-", "-"

    for cath_id in candidates:
        url = f"{CATH_API_BASE}/domain_summary/{cath_id}"
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                body = _fetch_json(url)
                if body is None:
                    break
                cathcode = _extract_cathcode(body)
                if cathcode != "-":
                    parts = [p for p in cathcode.split(".") if p]
                    level = "H" if len(parts) >= 4 else "T"
                    return cathcode, level, "cath_database"
                return "-", "-", "-"
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    break  # try next candidate ID
                last_exc = exc
            except Exception as exc:
                last_exc = exc
            if attempt + 1 < retries:
                time.sleep(delay * (attempt + 1))

    return "-", "-", "-"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def unique_ordered(items: list[str]) -> list[str]:
    """Return deduplicated list preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_true_dnames(raw: str) -> list[str]:
    """
    Parse true_dnames from raw CHAINSAW CSV into unique domain names.

    '4w7sA_d1|4w7sA_d2'                               → ['4w7sA_d1', '4w7sA_d2']
    '1m93B_d1|1m93B_d1|1m93B_d1|1m93B_d2|1m93B_d2'  → ['1m93B_d1', '1m93B_d2']
    """
    if not raw or not raw.strip():
        return []
    raw_names = [d.strip() for d in raw.split("|") if d.strip()]
    return unique_ordered(raw_names)


def build_enriched_label(chopping_star: str, per_domain_cath: list[str]) -> str:
    """
    Rebuild a TED-format label with real CATH superfamily codes.

    Input:  '1-260 | - * 261-443 | -',  ['3.40.50.300', '1.10.10.10']
    Output: '1-260 | 3.40.50.300 * 261-443 | 1.10.10.10'
    """
    domain_tokens = [t.strip() for t in DOMAIN_SPLIT_RE.split(chopping_star.strip()) if t.strip()]
    if len(domain_tokens) != len(per_domain_cath):
        # Domain count mismatch – keep original to avoid corrupting data
        return chopping_star
    rebuilt = []
    for token, cath in zip(domain_tokens, per_domain_cath):
        if "|" in token:
            boundary = token.split("|", 1)[0].strip()
            rebuilt.append(f"{boundary} | {cath}")
        else:
            rebuilt.append(token)
    return " * ".join(rebuilt)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw_dnames(raw_csv: Path) -> dict[str, list[str]]:
    """
    Read the raw CHAINSAW CSV and return {chain_id: [unique_domain_names]}.
    """
    mapping: dict[str, list[str]] = {}
    with raw_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            chain_id = (row.get("chain_id") or "").strip()
            raw_dnames = (row.get("true_dnames") or "").strip()
            if chain_id:
                mapping[chain_id] = parse_true_dnames(raw_dnames)
    return mapping


def collect_all_domain_names(dnames_map: dict[str, list[str]]) -> list[str]:
    """Collect every unique domain name across all chains."""
    seen: set[str] = set()
    out: list[str] = []
    for names in dnames_map.values():
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def resolve_cath_labels(
    all_domain_names: list[str],
    cache: dict,
    max_workers: int,
    retries: int,
    delay: float,
) -> dict[str, tuple[str, str, str]]:
    """
    Fetch CATH info for every domain name not already in cache.
    Returns {domain_name: (cathcode, level, method)}.
    """
    to_fetch = [n for n in all_domain_names if n not in cache]
    print(f"[INFO] domains in cache: {len(cache)}, to fetch: {len(to_fetch)}")

    results: dict[str, tuple[str, str, str]] = dict(cache)

    if not to_fetch:
        return results

    done = 0

    def _fetch(name: str):
        return name, fetch_cath_superfamily(name, retries=retries, delay=delay)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, name): name for name in to_fetch}
        for future in as_completed(futures):
            name, info = future.result()
            results[name] = info
            cache[name] = info
            done += 1
            if done % 50 == 0 or done == len(to_fetch):
                print(f"[INFO] fetched {done}/{len(to_fetch)}", flush=True)

    return results


def enrich_csv(
    input_csv: Path,
    out_csv: Path,
    dnames_map: dict[str, list[str]],
    cath_info: dict[str, tuple[str, str, str]],
    dry_run: bool,
    limit: int,
) -> dict:
    """
    Read input_csv, add CATH columns, write enriched out_csv.
    Returns summary counts.
    """
    stats = {
        "rows_total": 0,
        "rows_with_cath": 0,
        "rows_without_cath": 0,
        "domain_count_mismatch": 0,
        "cath_resolved": 0,
        "cath_missing": 0,
    }

    extra_fields = ["cath_labels", "cath_assignment_levels", "cath_assignment_methods"]

    rows_out = []
    fieldnames_out = None

    with input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames_out = (reader.fieldnames or []) + extra_fields

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            chain_id = (row.get("target_id") or "").strip()
            chopping_star = (row.get("chopping_star") or "").strip()

            domain_names = dnames_map.get(chain_id, [])
            n_domains_label = len(
                [t for t in DOMAIN_SPLIT_RE.split(chopping_star) if t.strip()]
            )

            if domain_names and len(domain_names) != n_domains_label:
                stats["domain_count_mismatch"] += 1
                # Fall back: use '-' for all domains
                domain_names = []

            # Gather per-domain CATH info
            cathcodes, levels, methods = [], [], []
            all_resolved = False
            if domain_names:
                for dname in domain_names:
                    code, level, method = cath_info.get(dname, ("-", "-", "-"))
                    cathcodes.append(code)
                    levels.append(level)
                    methods.append(method)
                    if code != "-":
                        stats["cath_resolved"] += 1
                    else:
                        stats["cath_missing"] += 1
                all_resolved = any(c != "-" for c in cathcodes)
            else:
                cathcodes = ["-"] * n_domains_label
                levels = ["-"] * n_domains_label
                methods = ["-"] * n_domains_label
                stats["cath_missing"] += n_domains_label

            # Rebuild TED-format labels with CATH codes
            enriched_label = build_enriched_label(chopping_star, cathcodes)
            row["label"] = enriched_label
            row["chopping_star"] = enriched_label

            row["cath_labels"] = "|".join(cathcodes)
            row["cath_assignment_levels"] = "|".join(levels)
            row["cath_assignment_methods"] = "|".join(methods)

            rows_out.append(row)
            stats["rows_total"] += 1
            if all_resolved:
                stats["rows_with_cath"] += 1
            else:
                stats["rows_without_cath"] += 1

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(rows_out)

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_csv", type=Path, default=RAW_CSV)
    parser.add_argument("--input_csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel CATH API workers (default 4)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Base delay between retries in seconds")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only first 10 rows (no API calls if cached)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N rows (0 = all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "chainsaw_cath1363_with_cath_labels.csv"
    summary_json = args.out_dir / "chainsaw_cath1363_cath_labels_summary.json"
    cache_path = args.out_dir / ".cath_label_cache.json"

    limit = 10 if args.dry_run else (args.limit or 0)

    # ------------------------------------------------------------------ #
    # Step 1: copy existing processed files into new directory             #
    # ------------------------------------------------------------------ #
    src_dir = args.input_csv.parent
    for src_file in src_dir.iterdir():
        dst_file = args.out_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"[INFO] copied {src_file.name} → {args.out_dir.name}/")

    # ------------------------------------------------------------------ #
    # Step 2: load raw CHAINSAW CSV → {chain_id: [domain_names]}          #
    # ------------------------------------------------------------------ #
    print(f"[INFO] loading raw CHAINSAW CSV: {args.raw_csv}")
    dnames_map = load_raw_dnames(args.raw_csv)
    print(f"[INFO] chains with domain names: {len(dnames_map)}")

    # ------------------------------------------------------------------ #
    # Step 3: resolve CATH labels via API (with local cache)               #
    # ------------------------------------------------------------------ #
    cache = load_cache(cache_path)
    all_domain_names = collect_all_domain_names(dnames_map)
    if limit:
        # Only fetch domains for the rows we'll actually process
        limited_chains: set[str] = set()
        with args.input_csv.open(newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if i >= limit:
                    break
                limited_chains.add((row.get("target_id") or "").strip())
        all_domain_names = [
            n for chain in limited_chains for n in dnames_map.get(chain, [])
        ]

    print(f"[INFO] unique domain names to resolve: {len(all_domain_names)}")
    cath_info = resolve_cath_labels(
        all_domain_names, cache,
        max_workers=args.max_workers,
        retries=args.retries,
        delay=args.delay,
    )
    save_cache(cache, cache_path)
    print(f"[INFO] cache saved to {cache_path}")

    # ------------------------------------------------------------------ #
    # Step 4: enrich the processed CSV and write output                    #
    # ------------------------------------------------------------------ #
    print(f"[INFO] enriching {args.input_csv.name} → {out_csv.name}")
    stats = enrich_csv(
        input_csv=args.input_csv,
        out_csv=out_csv,
        dnames_map=dnames_map,
        cath_info=cath_info,
        dry_run=args.dry_run,
        limit=limit,
    )

    # ------------------------------------------------------------------ #
    # Step 5: write summary                                                #
    # ------------------------------------------------------------------ #
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": str(args.input_csv),
        "raw_csv": str(args.raw_csv),
        "out_csv": str(out_csv),
        "dry_run": args.dry_run,
        "limit": limit,
        "cath_api_base": CATH_API_BASE,
        "cath_assignment_method": "cath_database",
        "stats": stats,
        "notes": (
            "CATH labels are fetched from the CATH v4.3 REST API using the domain names "
            "in the raw CHAINSAW CSV (true_dnames column). Domain names like '4w7sA_d1' "
            "are converted to CATH domain IDs (0-based '4w7sA00' then 1-based '4w7sA01'). "
            "The TED label format '<start>-<end> | <cath_label> * ...' is preserved; "
            "'-' placeholders are replaced with actual CATH superfamily codes."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\n[DONE]")
    print(json.dumps(stats, indent=2))
    print(f"\nOutput CSV:     {out_csv}")
    print(f"Summary:        {summary_json}")


if __name__ == "__main__":
    main()
