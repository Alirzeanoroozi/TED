
import os
from pathlib import Path

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAINS_CSV = REPO_ROOT / "data" / "chains.ted.csv"
TED_SEQ_ROOTS = [REPO_ROOT / "data" / "TED_seqA", REPO_ROOT / "data" / "TED_seqB"]
PARQUET_DIR = REPO_ROOT / "data" / "parquet_sequences"

CHUNK_SIZE = 10_000

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["uniprot_id"].astype(str), df["chopping_star"].astype(str)))

def read_fasta_sequence(fasta_path):
    lines = Path(fasta_path).read_text().splitlines()
    seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
    return "".join(seq_lines)

def iter_fasta_files(_roots):
    fasta_files = []
    for root in TED_SEQ_ROOTS:
        if not root.exists():
            raise FileNotFoundError(f"Missing FASTA directory: {root}")
        for file in os.listdir(root):
            if file.endswith(".fasta"):
                fasta_files.append(os.path.join(root, file))
    return sorted(fasta_files)


def write_parquet_frame(df, out_path):
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as exc:
        if pl is None:
            raise ImportError(
                "Writing parquet requires pandas parquet support (pyarrow/fastparquet) "
                "or an installed polars fallback."
            ) from exc
        pl.DataFrame(df.to_dict(orient="records")).write_parquet(str(out_path))

def write_parquet_chunk(rows, chunk_idx):
    if not rows:
        return
    os.makedirs(PARQUET_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = PARQUET_DIR / f"ted_sequences_{chunk_idx:05d}.parquet"
    write_parquet_frame(df, out_path)
    print(f"[INFO] wrote {len(df)} rows to {out_path}")

def create_parquet_shards():
    labels = load_labels(CHAINS_CSV)
    fasta_files = iter_fasta_files(TED_SEQ_ROOTS)
    print(len(labels), len(fasta_files))
    
    buffer = []
    chunk_idx = 0

    for fasta_path in fasta_files:
        uniprot_id = os.path.basename(fasta_path).split(".")[0]
        label = labels.get(uniprot_id)
        if label is None:
            continue

        sequence = read_fasta_sequence(fasta_path)
        buffer.append(
            {
                "uniprot_id": uniprot_id,
                "sequence": sequence,
                "label": label,
            }
        )

        if len(buffer) >= CHUNK_SIZE:
            chunk_idx += 1
            write_parquet_chunk(buffer, chunk_idx)
            buffer.clear()

    if buffer:
        chunk_idx += 1
        write_parquet_chunk(buffer, chunk_idx)


if __name__ == "__main__":
    create_parquet_shards()
