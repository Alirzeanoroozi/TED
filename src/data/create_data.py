
import pandas as pd
import os

CHAINS_CSV = "data/chains.ted.csv"
TED_SEQ_ROOTS = ["data/TED_sequences/TED_seqA", "data/TED_sequences/TED_seqB"]
PARQUET_DIR = "data/parquet_sequences"

CHUNK_SIZE = 10_000

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["uniprot_id"].astype(str), df["chopping_star"].astype(str)))

def read_fasta_sequence(fasta_path):
    lines = open(fasta_path, "r").read().splitlines()
    seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
    return "".join(seq_lines)

def iter_fasta_files(root):
    fasta_files = []
    for root in TED_SEQ_ROOTS:
        for file in os.listdir(root):
            if file.endswith(".fasta"):
                fasta_files.append(os.path.join(root, file))
    return fasta_files

def write_parquet_chunk(rows, chunk_idx):
    if not rows:
        return
    os.makedirs(PARQUET_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(PARQUET_DIR, f"ted_sequences_{chunk_idx:05d}.parquet")
    df.to_parquet(out_path, index=False)

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
