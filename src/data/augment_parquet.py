"""
Augment TED base parquet shards by cropping sequences only at valid inter-domain cut points.

Example on the cluster from /scratch/erkmenerken22/TED:
  source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh && conda activate ted
  python src/data/augment_parquet.py --in_dir data/parquet_sequences --out_dir data/parquet_augmented --n_aug_per_chain 5 --seed 42
"""

import argparse
import random
import re
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "augment_parquet.py requires pandas. "
        "Activate the TED environment first, for example: "
        "source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh && "
        "conda activate gmconda_py3923"
    ) from exc

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

DOMAIN_SPLIT_RE = re.compile(r"\s*\*\s*")
SEGMENT_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
DEFAULT_PROGRESS_EVERY = 1000
DEFAULT_DRY_RUN_ROWS = 5


def detect_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "data").exists() and (candidate / "src" / "data").exists():
            return candidate
    raise RuntimeError("Could not detect the TED repo root from src/data/augment_parquet.py")


def resolve_path(raw_path: Union[str, Path], repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def detect_input_dir(repo_root: Path) -> Path:
    preferred_dirs = [repo_root / "data" / "parquet_sequences", repo_root / "data" / "parquet_base"]
    for path in preferred_dirs:
        if path.is_dir():
            return path

    parquet_files = sorted((repo_root / "data").rglob("*.parquet"))
    if parquet_files:
        return parquet_files[0].parent
    return preferred_dirs[0]


def list_parquet_files(in_dir: Path) -> List[Path]:
    if in_dir.is_file():
        if in_dir.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not parquet: {in_dir}")
        return [in_dir]
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
    if not in_dir.is_dir():
        raise ValueError(f"Input path is neither a parquet file nor a directory: {in_dir}")
    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files were found in {in_dir}. "
            "Expected base shards under data/parquet_sequences or pass --in_dir explicitly."
        )
    return parquet_files


def read_parquet_frame(parquet_file: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(parquet_file)
    except ImportError as exc:
        if pl is None:
            raise ImportError(
                "Reading parquet requires pandas parquet support (pyarrow/fastparquet) "
                "or an installed polars fallback."
            ) from exc
        return pd.DataFrame(pl.read_parquet(parquet_file).to_dicts())


def write_parquet_frame(df: pd.DataFrame, out_path: Path) -> None:
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as exc:
        if pl is None:
            raise ImportError(
                "Writing parquet requires pandas parquet support (pyarrow/fastparquet) "
                "or an installed polars fallback."
            ) from exc
        pl.DataFrame(df.to_dict(orient="records")).write_parquet(str(out_path))


def detect_target_column(columns: Iterable[str]) -> str:
    column_set = set(columns)
    if "chopping_star" in column_set:
        return "chopping_star"
    if "label" in column_set:
        return "label"
    raise ValueError("Expected a domain annotation column named 'chopping_star' or 'label'.")


def split_domain_tokens(chopping_star: object) -> List[str]:
    if chopping_star is None or pd.isna(chopping_star):
        return []
    text = str(chopping_star).strip()
    if not text:
        return []
    return [token.strip() for token in DOMAIN_SPLIT_RE.split(text) if token.strip()]


def infer_index_base(parquet_files: List[Path], dry_limit_rows: int = 200) -> int:
    starts: List[int] = []
    for parquet_file in parquet_files[:5]:
        df = read_parquet_frame(parquet_file)
        target_column = detect_target_column(df.columns)
        for value in df[target_column].head(dry_limit_rows).tolist():
            for token in split_domain_tokens(value):
                left = token.split("|", 1)[0].strip()
                for segment in [seg.strip() for seg in left.split("_") if seg.strip()]:
                    match = SEGMENT_RE.match(segment)
                    if match:
                        starts.append(int(match.group(1)))
                        if len(starts) >= dry_limit_rows:
                            break
                if len(starts) >= dry_limit_rows:
                    break
            if len(starts) >= dry_limit_rows:
                break
        if len(starts) >= dry_limit_rows:
            break

    if any(start == 0 for start in starts):
        return 0
    return 1


def parse_annotation(annotation: object, index_base: int) -> List[Dict[str, object]]:
    domains: List[Dict[str, object]] = []
    for token in split_domain_tokens(annotation):
        if "|" not in token:
            raise ValueError(f"Domain token is missing '|': {token}")
        left, right = token.split("|", 1)
        cath_label = right.strip()
        segments: List[Tuple[int, int]] = []
        for segment_text in [seg.strip() for seg in left.strip().split("_") if seg.strip()]:
            match = SEGMENT_RE.match(segment_text)
            if not match:
                raise ValueError(f"Malformed segment in annotation: {segment_text}")
            start_text, end_text = int(match.group(1)), int(match.group(2))
            if end_text < start_text:
                raise ValueError(f"Segment end precedes start: {segment_text}")
            start = start_text - index_base
            end = end_text - index_base + 1
            segments.append((start, end))

        segments.sort()
        if not segments:
            continue
        domains.append(
            {
                "segments": segments,
                "cath_label": cath_label,
                "start": min(start for start, _ in segments),
                "end": max(end for _, end in segments),
            }
        )
    return domains


def serialize_domains(domains: List[Dict[str, object]], index_base: int) -> str:
    tokens: List[str] = []
    for domain in domains:
        segment_tokens = []
        for start, end in domain["segments"]:
            start_out = start + index_base
            end_out = end - 1 + index_base
            segment_tokens.append(f"{start_out}-{end_out}")
        tokens.append(f"{'_'.join(segment_tokens)} | {domain['cath_label']}")
    return " * ".join(tokens)


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_valid_cut_points(seq_len: int, domains: List[Dict[str, object]]) -> List[int]:
    # Protect the whole domain envelope so we never cut between segments of the same domain.
    protected = merge_intervals([(int(domain["start"]), int(domain["end"])) for domain in domains])
    valid_points: List[int] = []
    interval_idx = 0
    for point in range(seq_len + 1):
        while interval_idx < len(protected) and point >= protected[interval_idx][1]:
            interval_idx += 1
        if interval_idx < len(protected):
            start, end = protected[interval_idx]
            if start < point < end:
                continue
        valid_points.append(point)
    return valid_points


def collect_random_windows(
    valid_points: List[int],
    seq_len: int,
    min_len: int,
    max_len: Optional[int],
    n_windows: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    max_right_limit = seq_len if max_len is None else min(seq_len, max_len)
    if len(valid_points) < 2:
        return windows

    max_attempts = max(50, n_windows * 40)
    for _ in range(max_attempts):
        left_index = rng.randrange(0, len(valid_points) - 1)
        left_cut = valid_points[left_index]
        lo = bisect_left(valid_points, left_cut + min_len, lo=left_index + 1)
        hi = bisect_right(valid_points, left_cut + max_right_limit, lo=lo)
        if lo >= hi:
            continue
        right_cut = valid_points[rng.randrange(lo, hi)]
        if left_cut == 0 and right_cut == seq_len:
            continue
        window = (left_cut, right_cut)
        if window in seen:
            continue
        seen.add(window)
        windows.append(window)
        if len(windows) >= n_windows:
            break
    return windows


def evenly_spaced_indices(size: int, count: int) -> List[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [size // 2]
    indices = []
    for idx in range(count):
        pos = round(idx * (size - 1) / (count - 1))
        if not indices or pos != indices[-1]:
            indices.append(pos)
    return indices


def collect_systematic_windows(
    valid_points: List[int],
    seq_len: int,
    min_len: int,
    max_len: Optional[int],
    n_windows: int,
) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    max_right_limit = seq_len if max_len is None else min(seq_len, max_len)
    left_indices = evenly_spaced_indices(max(len(valid_points) - 1, 0), max(n_windows * 3, 1))

    for left_index in left_indices:
        if left_index >= len(valid_points) - 1:
            continue
        left_cut = valid_points[left_index]
        lo = bisect_left(valid_points, left_cut + min_len, lo=left_index + 1)
        hi = bisect_right(valid_points, left_cut + max_right_limit, lo=lo)
        if lo >= hi:
            continue
        candidate_indices = [hi - 1, (lo + hi - 1) // 2, lo]
        for right_index in candidate_indices:
            right_cut = valid_points[right_index]
            if left_cut == 0 and right_cut == seq_len:
                continue
            window = (left_cut, right_cut)
            if window in seen:
                continue
            seen.add(window)
            windows.append(window)
            if len(windows) >= n_windows:
                return windows

    for left_index, left_cut in enumerate(valid_points[:-1]):
        lo = bisect_left(valid_points, left_cut + min_len, lo=left_index + 1)
        hi = bisect_right(valid_points, left_cut + max_right_limit, lo=lo)
        if lo >= hi:
            continue
        right_cut = valid_points[hi - 1]
        if left_cut == 0 and right_cut == seq_len:
            continue
        window = (left_cut, right_cut)
        if window in seen:
            continue
        seen.add(window)
        windows.append(window)
        if len(windows) >= n_windows:
            break

    return windows


def select_windows(
    valid_points: List[int],
    seq_len: int,
    min_len: int,
    max_len: Optional[int],
    n_windows: int,
    strategy: str,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if strategy == "random":
        return collect_random_windows(valid_points, seq_len, min_len, max_len, n_windows, rng)
    return collect_systematic_windows(valid_points, seq_len, min_len, max_len, n_windows)


def crop_domains(
    domains: List[Dict[str, object]],
    left_cut: int,
    right_cut: int,
) -> Tuple[List[Dict[str, object]], int, int]:
    kept_domains: List[Dict[str, object]] = []
    kept_count = 0
    dropped_partial = 0

    for domain in domains:
        segments = domain["segments"]
        fully_contained = all(left_cut <= start and end <= right_cut for start, end in segments)
        partial_overlap = any((start < left_cut < end) or (start < right_cut < end) for start, end in segments)
        any_overlap = any(not (end <= left_cut or start >= right_cut) for start, end in segments)

        if fully_contained:
            translated_segments = [(start - left_cut, end - left_cut) for start, end in segments]
            kept_domains.append({"segments": translated_segments, "cath_label": domain["cath_label"]})
            kept_count += 1
        elif partial_overlap or any_overlap:
            dropped_partial += 1

    return kept_domains, kept_count, dropped_partial


def validate_window(left_cut: int, right_cut: int, valid_points: Set[int]) -> None:
    if left_cut >= right_cut:
        raise ValueError(f"Invalid crop window: left_cut={left_cut}, right_cut={right_cut}")
    if left_cut not in valid_points or right_cut not in valid_points:
        raise ValueError(f"Crop boundary is not a valid cut point: ({left_cut}, {right_cut})")


def build_original_row(row: Dict[str, object], target_column: str) -> Dict[str, object]:
    output = dict(row)
    annotation = str(output.get(target_column, ""))
    output["label"] = annotation
    output["chopping_star"] = annotation
    output["is_augmented"] = False
    output["augmented_from"] = None
    output["left_cut"] = 0
    output["right_cut"] = len(str(output.get("sequence", "")))
    return output


def build_augmented_row(
    row: Dict[str, object],
    row_key: str,
    augmented_id: int,
    cropped_seq: str,
    updated_annotation: str,
    left_cut: int,
    right_cut: int,
) -> Dict[str, object]:
    output = dict(row)
    original_id = str(output.get("uniprot_id", row_key))
    output["sequence"] = cropped_seq
    output["label"] = updated_annotation
    output["chopping_star"] = updated_annotation
    if "uniprot_id" in output:
        output["uniprot_id"] = f"{original_id}__aug_{augmented_id:02d}"
    output["is_augmented"] = True
    output["augmented_from"] = original_id
    output["left_cut"] = left_cut
    output["right_cut"] = right_cut
    return output


def row_key_from(row: Dict[str, object], shard_name: str, row_index: int) -> str:
    if "uniprot_id" in row and pd.notna(row["uniprot_id"]):
        return str(row["uniprot_id"])
    return f"{shard_name}:{row_index}"


def iter_output_chunks(rows: List[Dict[str, object]], shard_size: Optional[int]) -> Iterable[List[Dict[str, object]]]:
    if shard_size is None or shard_size <= 0:
        yield rows
        return
    for start in range(0, len(rows), shard_size):
        yield rows[start : start + shard_size]


def write_output_shards(
    output_rows: List[Dict[str, object]],
    out_dir: Path,
    input_stem: str,
    shard_size: Optional[int],
) -> List[Path]:
    written_paths: List[Path] = []
    for chunk_index, chunk_rows in enumerate(iter_output_chunks(output_rows, shard_size), start=1):
        if not chunk_rows:
            continue
        suffix = "" if shard_size is None else f"_{chunk_index:05d}"
        out_path = out_dir / f"{input_stem}_augmented{suffix}.parquet"
        write_parquet_frame(pd.DataFrame(chunk_rows), out_path)
        written_paths.append(out_path)
    return written_paths


def process_row(
    row: Dict[str, object],
    target_column: str,
    index_base: int,
    n_aug_per_chain: int,
    min_len: int,
    max_len: Optional[int],
    strategy: str,
    allow_no_domains: bool,
    include_original: bool,
    row_seed: int,
    shard_name: str,
    row_index: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[str]]:
    stats = {
        "augmented_rows": 0,
        "kept_domains": 0,
        "dropped_domains": 0,
        "skipped_no_domains": 0,
        "parse_errors": 0,
        "candidate_windows": 0,
    }
    warnings: List[str] = []
    output_rows: List[Dict[str, object]] = []

    sequence = str(row.get("sequence", "") or "")
    annotation = row.get(target_column, "")
    row_key = row_key_from(row, shard_name, row_index)

    if include_original:
        output_rows.append(build_original_row(row, target_column))

    if not sequence:
        return output_rows, stats, warnings

    try:
        domains = parse_annotation(annotation, index_base)
    except ValueError as exc:
        stats["parse_errors"] += 1
        warnings.append(f"{row_key}: parse_error: {exc}")
        return output_rows, stats, warnings

    seq_len = len(sequence)
    if max_len is not None and max_len < min_len:
        raise ValueError(f"max_len ({max_len}) cannot be smaller than min_len ({min_len})")

    valid_points = compute_valid_cut_points(seq_len, domains)
    valid_point_set = set(valid_points)
    rng = random.Random(row_seed)
    windows = select_windows(valid_points, seq_len, min_len, max_len, n_aug_per_chain, strategy, rng)
    stats["candidate_windows"] += len(windows)

    for augmented_id, (left_cut, right_cut) in enumerate(windows, start=1):
        validate_window(left_cut, right_cut, valid_point_set)
        cropped_seq = sequence[left_cut:right_cut]
        kept_domains, kept_count, dropped_partial = crop_domains(domains, left_cut, right_cut)
        stats["kept_domains"] += kept_count
        stats["dropped_domains"] += dropped_partial

        if not kept_domains and not allow_no_domains:
            stats["skipped_no_domains"] += 1
            continue

        updated_annotation = serialize_domains(kept_domains, index_base) if kept_domains else ""

        for domain in kept_domains:
            for start, end in domain["segments"]:
                if not (0 <= start < end <= len(cropped_seq)):
                    raise ValueError(
                        f"Translated domain out of range for {row_key}: {(start, end)} vs cropped length {len(cropped_seq)}"
                    )

        output_rows.append(
            build_augmented_row(
                row=row,
                row_key=row_key,
                augmented_id=augmented_id,
                cropped_seq=cropped_seq,
                updated_annotation=updated_annotation,
                left_cut=left_cut,
                right_cut=right_cut,
            )
        )
        stats["augmented_rows"] += 1

    return output_rows, stats, warnings


def process_parquet_shard(
    parquet_file: Path,
    out_dir: Path,
    index_base: int,
    seed: int,
    n_aug_per_chain: int,
    min_len: int,
    max_len: Optional[int],
    strategy: str,
    include_original: bool,
    allow_no_domains: bool,
    shard_size: Optional[int],
    progress_every: int,
    global_counts: Dict[str, int],
    start_time: float,
) -> List[Path]:
    df = read_parquet_frame(parquet_file)
    target_column = detect_target_column(df.columns)
    records = df.to_dict(orient="records")
    output_rows: List[Dict[str, object]] = []
    shard_warnings: List[str] = []

    for row_index, row in enumerate(records, start=1):
        row_seed = seed + global_counts["rows_processed"]
        row_output_rows, row_stats, row_warnings = process_row(
            row=row,
            target_column=target_column,
            index_base=index_base,
            n_aug_per_chain=n_aug_per_chain,
            min_len=min_len,
            max_len=max_len,
            strategy=strategy,
            allow_no_domains=allow_no_domains,
            include_original=include_original,
            row_seed=row_seed,
            shard_name=parquet_file.stem,
            row_index=row_index,
        )
        output_rows.extend(row_output_rows)
        shard_warnings.extend(row_warnings)
        global_counts["rows_processed"] += 1
        global_counts["augmented_rows"] += row_stats["augmented_rows"]
        global_counts["kept_domains"] += row_stats["kept_domains"]
        global_counts["dropped_domains"] += row_stats["dropped_domains"]
        global_counts["skipped_no_domains"] += row_stats["skipped_no_domains"]
        global_counts["parse_errors"] += row_stats["parse_errors"]

        if global_counts["rows_processed"] % progress_every == 0:
            elapsed = time.time() - start_time
            print(
                "[PROGRESS] "
                f"rows={global_counts['rows_processed']:,} "
                f"augmented_rows={global_counts['augmented_rows']:,} "
                f"kept_domains={global_counts['kept_domains']:,} "
                f"dropped_domains={global_counts['dropped_domains']:,} "
                f"skipped_no_domains={global_counts['skipped_no_domains']:,} "
                f"parse_errors={global_counts['parse_errors']:,} "
                f"elapsed_sec={elapsed:.1f}"
            )

    for warning in shard_warnings[:10]:
        print(f"[WARN] {warning}")
    if len(shard_warnings) > 10:
        print(f"[WARN] Suppressed {len(shard_warnings) - 10} additional parse warnings for {parquet_file.name}")

    return write_output_shards(output_rows, out_dir, parquet_file.stem, shard_size)


def run_dry_run(
    parquet_files: List[Path],
    index_base: int,
    seed: int,
    n_aug_per_chain: int,
    min_len: int,
    max_len: Optional[int],
    strategy: str,
    include_original: bool,
    allow_no_domains: bool,
) -> None:
    shown = 0
    for parquet_file in parquet_files:
        df = read_parquet_frame(parquet_file)
        target_column = detect_target_column(df.columns)
        for row_index, row in enumerate(df.to_dict(orient="records"), start=1):
            row_output_rows, row_stats, row_warnings = process_row(
                row=row,
                target_column=target_column,
                index_base=index_base,
                n_aug_per_chain=n_aug_per_chain,
                min_len=min_len,
                max_len=max_len,
                strategy=strategy,
                allow_no_domains=allow_no_domains,
                include_original=include_original,
                row_seed=seed + shown,
                shard_name=parquet_file.stem,
                row_index=row_index,
            )
            print(f"[DRY RUN] source_shard={parquet_file.name} row={row_index} id={row_key_from(row, parquet_file.stem, row_index)}")
            print(f"[DRY RUN] original_len={len(str(row.get('sequence', '') or ''))} annotation={row.get(target_column, '')}")
            for warning in row_warnings:
                print(f"[DRY RUN] warning={warning}")
            augmented_examples = [item for item in row_output_rows if item.get("is_augmented")]
            for example in augmented_examples:
                print(
                    "[DRY RUN] augmented "
                    f"id={example.get('uniprot_id')} len={len(str(example.get('sequence', '')))} "
                    f"cuts=({example.get('left_cut')}, {example.get('right_cut')}) "
                    f"annotation={example.get('chopping_star', '')}"
                )
            if not augmented_examples:
                print(f"[DRY RUN] no augmented samples kept; stats={row_stats}")
            print()
            shown += 1
            if shown >= DEFAULT_DRY_RUN_ROWS:
                return


def build_arg_parser(repo_root: Path, detected_in_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment TED parquet shards by cropping in valid inter-domain regions.")
    parser.add_argument(
        "--in_dir",
        default=str(detected_in_dir),
        help=f"Input parquet directory or file. Default auto-detected: {detected_in_dir}",
    )
    parser.add_argument(
        "--out_dir",
        default=str(repo_root / "data" / "parquet_sequences_augmented"),
        help=f"Output directory for augmented parquet shards. Default: {repo_root / 'data' / 'parquet_sequences_augmented'}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for crop selection.")
    parser.add_argument("--n_aug_per_chain", type=int, default=3, help="Number of augmented crops to attempt per chain.")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum cropped sequence length.")
    parser.add_argument("--max_len", type=int, default=None, help="Optional maximum cropped sequence length.")
    parser.add_argument("--strategy", choices=["random", "systematic"], default="random", help="Crop selection strategy.")
    parser.add_argument("--allow_no_domains", action="store_true", help="Keep augmented crops even if no domain remains after cropping.")
    parser.add_argument("--shard_size", type=int, default=None, help="Optional max output rows per written parquet shard.")
    parser.add_argument("--dry_run", action="store_true", help="Process about five rows and print before/after examples without writing parquet.")
    parser.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY, help=f"Progress log cadence in rows. Default: {DEFAULT_PROGRESS_EVERY}")
    parser.set_defaults(include_original=True)
    parser.add_argument("--include_original", dest="include_original", action="store_true", help="Keep original rows in the output dataset (default).")
    parser.add_argument("--no_include_original", dest="include_original", action="store_false", help="Exclude original rows and write only augmented rows.")
    return parser


def main() -> None:
    repo_root = detect_repo_root()
    detected_in_dir = detect_input_dir(repo_root)
    parser = build_arg_parser(repo_root, detected_in_dir)
    args = parser.parse_args()

    in_dir = resolve_path(args.in_dir, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)

    if in_dir == out_dir:
        raise ValueError(f"Output directory must differ from input directory: {in_dir}")

    parquet_files = list_parquet_files(in_dir)
    index_base = infer_index_base(parquet_files)

    print(f"[INFO] detected_in_dir={in_dir}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] parquet_shards={len(parquet_files)}")
    print(f"[INFO] inferred_index_base={index_base}")

    if args.dry_run:
        run_dry_run(
            parquet_files=parquet_files,
            index_base=index_base,
            seed=args.seed,
            n_aug_per_chain=args.n_aug_per_chain,
            min_len=args.min_len,
            max_len=args.max_len,
            strategy=args.strategy,
            include_original=args.include_original,
            allow_no_domains=args.allow_no_domains,
        )
        return

    if out_dir.exists() and any(out_dir.glob("*.parquet")):
        raise FileExistsError(
            f"Output directory already contains parquet files: {out_dir}. "
            "Choose a new --out_dir so the base dataset is not overwritten."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    global_counts = {
        "rows_processed": 0,
        "augmented_rows": 0,
        "kept_domains": 0,
        "dropped_domains": 0,
        "skipped_no_domains": 0,
        "parse_errors": 0,
    }
    start_time = time.time()
    written_files: List[Path] = []

    for parquet_file in parquet_files:
        print(f"[INFO] processing_shard={parquet_file}")
        written_files.extend(
            process_parquet_shard(
                parquet_file=parquet_file,
                out_dir=out_dir,
                index_base=index_base,
                seed=args.seed,
                n_aug_per_chain=args.n_aug_per_chain,
                min_len=args.min_len,
                max_len=args.max_len,
                strategy=args.strategy,
                include_original=args.include_original,
                allow_no_domains=args.allow_no_domains,
                shard_size=args.shard_size,
                progress_every=args.progress_every,
                global_counts=global_counts,
                start_time=start_time,
            )
        )

    elapsed = time.time() - start_time
    print(
        "[DONE] "
        f"rows_processed={global_counts['rows_processed']:,} "
        f"augmented_rows={global_counts['augmented_rows']:,} "
        f"kept_domains={global_counts['kept_domains']:,} "
        f"dropped_domains={global_counts['dropped_domains']:,} "
        f"skipped_no_domains={global_counts['skipped_no_domains']:,} "
        f"parse_errors={global_counts['parse_errors']:,} "
        f"written_files={len(written_files)} "
        f"elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
