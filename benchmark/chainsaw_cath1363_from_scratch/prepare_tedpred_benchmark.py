#!/usr/bin/env python3
"""Build a TEDPred-ready dataset from freshly downloaded CHAINSAW CATH1363 data.

This script intentionally does not read ``benchmark/results``.  It expects the
official CHAINSAW CSV downloaded into this folder's ``data/raw`` directory, then
downloads PDB files from RCSB and extracts the requested chain sequence from ATOM
records.  That keeps the TEDPred input sequence aligned to the PDB residue order
assumed by CHAINSAW's zero-indexed benchmark coordinates.
"""

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_CSV = SCRIPT_DIR / "data" / "raw" / "chainsaw_model_v3_on_cath1363_test.csv"
DEFAULT_STRUCTURES_DIR = SCRIPT_DIR / "data" / "structures" / "pdb"
DEFAULT_OUT_CSV = SCRIPT_DIR / "data" / "processed" / "chainsaw_cath1363_tedpred.csv"
DEFAULT_EXCLUDED_CSV = SCRIPT_DIR / "data" / "processed" / "chainsaw_cath1363_excluded.csv"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "data" / "processed" / "chainsaw_cath1363_summary.json"

RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified residues mapped to their parent amino acid.
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "CSO": "C",
    "HYP": "P",
    "MLZ": "K",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url, timeout, retries):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TEDPred CHAINSAW benchmark setup"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(5.0, 0.5 * (attempt + 1)))
    raise RuntimeError("failed to download {}: {}".format(url, last_error))


def download_pdb(pdb_id, structures_dir, timeout, retries, force):
    structures_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.lower()
    out_path = structures_dir / "{}.pdb".format(pdb_id)
    if out_path.exists() and not force:
        return out_path, "already_exists"

    url = PDB_URL.format(pdb_id=pdb_id.upper())
    data = fetch_bytes(url, timeout, retries)
    out_path.write_bytes(data)
    return out_path, "downloaded"


def split_chain_id(chain_id):
    chain_id = chain_id.strip()
    if len(chain_id) < 5:
        raise ValueError("expected chain_id like 4w7sA, got {!r}".format(chain_id))
    return chain_id[:4].lower(), chain_id[4:]


def parse_zero_based_chopping(chopping):
    domains = []
    for raw_domain in chopping.split("|"):
        raw_domain = raw_domain.strip()
        if not raw_domain:
            raise ValueError("empty domain")
        segments = []
        for raw_segment in raw_domain.split("_"):
            raw_segment = raw_segment.strip()
            match = RANGE_RE.match(raw_segment)
            if not match:
                raise ValueError("invalid segment {!r}".format(raw_segment))
            start, end = map(int, match.groups())
            if start > end:
                raise ValueError("descending segment {!r}".format(raw_segment))
            segments.append((start, end))
        domains.append(segments)
    return domains


def zero_based_to_ted_label(chopping):
    ted_domains = []
    for segments in parse_zero_based_chopping(chopping):
        converted = ["{}-{}".format(start + 1, end + 1) for start, end in segments]
        ted_domains.append("{} | -".format("_".join(converted)))
    return " * ".join(ted_domains)


def max_required_length(chopping):
    max_end = 0
    for domain in parse_zero_based_chopping(chopping):
        for _, end in domain:
            max_end = max(max_end, end + 1)
    return max_end


def _chain_matches(found_chain, wanted_chain):
    return found_chain == wanted_chain or found_chain.lower() == wanted_chain.lower()


def extract_pdb_chain_sequence(pdb_path, wanted_chain):
    residues = []
    seen = set()
    saw_model = False
    in_first_model = True

    with pdb_path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if saw_model:
                    in_first_model = False
                saw_model = True
                continue
            if record.startswith("ENDMDL") and saw_model:
                break
            if not in_first_model:
                continue
            if record not in {"ATOM  ", "HETATM"}:
                continue
            if len(line) < 27:
                continue

            chain = line[21].strip()
            if not _chain_matches(chain, wanted_chain):
                continue

            resname = line[17:20].strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                if record == "ATOM  ":
                    aa = "X"
                else:
                    continue

            residue_key = (chain, line[22:26].strip(), line[26].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            residues.append(aa)

    return "".join(residues)


def present(value):
    return value is not None and str(value).strip() not in {"", "nan", "NaN"}


def prepare(args):
    if not args.raw_csv.exists():
        raise SystemExit(
            "Missing raw CHAINSAW CSV: {}\nRun download_official_chainsaw.py first.".format(
                args.raw_csv
            )
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    included = []
    excluded = []
    pdb_downloads = Counter()
    domain_groups = Counter()

    with args.raw_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if args.limit and row_index > args.limit:
                break

            chain_id = (row.get("chain_id") or "").strip()
            try:
                pdb_id, chain = split_chain_id(chain_id)
            except ValueError as exc:
                excluded.append({"chain_id": chain_id, "reason": str(exc)})
                continue

            true_chopping = (row.get("true_chopping") or "").strip()
            n_domains_text = (row.get("n_domains") or "").strip()
            if not present(true_chopping) or not present(n_domains_text):
                excluded.append(
                    {
                        "chain_id": chain_id,
                        "reason": "missing_chainsaw_truth:{}".format(row.get("error") or ""),
                    }
                )
                continue

            try:
                n_domains = int(float(n_domains_text))
                label = zero_based_to_ted_label(true_chopping)
                nres_floor = max_required_length(true_chopping)
            except ValueError as exc:
                excluded.append({"chain_id": chain_id, "reason": "invalid_chopping:{}".format(exc)})
                continue

            try:
                pdb_path, status = download_pdb(
                    pdb_id,
                    args.structures_dir,
                    args.timeout,
                    args.retries,
                    args.force_download_structures,
                )
                pdb_downloads[status] += 1
            except RuntimeError as exc:
                excluded.append({"chain_id": chain_id, "reason": "pdb_download_failed:{}".format(exc)})
                continue

            sequence = extract_pdb_chain_sequence(pdb_path, chain)
            if not sequence:
                excluded.append({"chain_id": chain_id, "reason": "chain_not_found_in_pdb"})
                continue
            if len(sequence) < nres_floor:
                excluded.append(
                    {
                        "chain_id": chain_id,
                        "reason": "sequence_too_short:{}<{}".format(len(sequence), nres_floor),
                    }
                )
                continue

            domain_group = "single" if n_domains == 1 else "multi"
            domain_groups[domain_group] += 1
            included.append(
                {
                    "target_id": chain_id,
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "sequence": sequence,
                    "nres": len(sequence),
                    "chopping_star": label,
                    "label": label,
                    "n_domains": n_domains,
                    "domain_group": domain_group,
                    "chainsaw_true_chopping_zero_based": true_chopping,
                    "chainsaw_pred_chopping_zero_based": row.get("pred_chopping") or "",
                    "chainsaw_iou": row.get("iou") or "",
                    "chainsaw_correct_prop": row.get("proportion_correct_domains") or "",
                    "chainsaw_boundary_distance_score": row.get("boundary_dist_score") or "",
                    "source": "JudeWells/chainsaw:data_and_benchmarking/chainsaw_model_v3_on_cath1363_test.csv",
                    "sequence_source": "RCSB PDB ATOM records",
                }
            )

    fieldnames = [
        "target_id",
        "pdb_id",
        "chain",
        "sequence",
        "nres",
        "chopping_star",
        "label",
        "n_domains",
        "domain_group",
        "chainsaw_true_chopping_zero_based",
        "chainsaw_pred_chopping_zero_based",
        "chainsaw_iou",
        "chainsaw_correct_prop",
        "chainsaw_boundary_distance_score",
        "source",
        "sequence_source",
    ]
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(included)

    with args.excluded_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain_id", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "raw_csv": str(args.raw_csv),
        "raw_csv_sha256": sha256_file(args.raw_csv),
        "structures_dir": str(args.structures_dir),
        "out_csv": str(args.out_csv),
        "excluded_csv": str(args.excluded_csv),
        "rows_included": len(included),
        "rows_excluded": len(excluded),
        "domain_group_counts": dict(domain_groups),
        "pdb_downloads": dict(pdb_downloads),
        "indexing_note": "CHAINSAW ranges are zero-based inclusive; TEDPred labels are one-based inclusive.",
        "classification_note": "CATH labels are '-' because this CHAINSAW benchmark evaluates domain boundaries.",
    }

    if args.write_parquet:
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("--write_parquet requires pandas/pyarrow in the active env") from exc
        parquet_path = args.out_csv.with_suffix(".parquet")
        pd.read_csv(args.out_csv).to_parquet(parquet_path, index=False)
        summary["out_parquet"] = str(parquet_path)

    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--structures_dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--excluded_csv", type=Path, default=DEFAULT_EXCLUDED_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force_download_structures", action="store_true")
    parser.add_argument("--write_parquet", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
