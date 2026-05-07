import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from pathlib import Path
import numpy as np

def resolve_parquet_inputs(data_parquet_path: str) -> list[str]:
    parquet_files = sorted(str(parquet_file) for parquet_file in Path(data_parquet_path).glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No .parquet files found under {data_parquet_path}")
    return parquet_files

def load_file(path):
    df = pd.read_parquet(path)
    if "chopping_star" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "chopping_star"})
    return df

def load_data(paths):
    dfs = [load_file(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["sequence", "chopping_star"])

def split_datasets(paths, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, val_size=1000, test_size=100):
    df = load_data(paths)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
    n = len(df)
    train_df = df.iloc[:n - val_size - test_size]
    val_df = df.iloc[n - val_size - test_size : n - test_size]
    test_df = df.iloc[n - test_size : n]

    train_dataset = SequenceToChoppingDataset(src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=train_df)
    val_dataset = SequenceToChoppingDataset(src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=val_df)
    test_dataset = SequenceToChoppingDataset(src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=test_df)
    return train_dataset, val_dataset, test_dataset

class SequenceToChoppingDataset(Dataset):
    def __init__(self, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=None):
        self.df = df
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row["sequence"]).strip()[: self.max_src_len]
        tgt_text = str(row["chopping_star"]).strip()[: self.max_tgt_len]

        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)

        src = torch.tensor(src_ids, dtype=torch.long)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        tgt_in = tgt[:-1].clone()
        tgt_out = tgt[1:].clone()

        return {
            "src": src,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "src_len": src.size(0),
            "tgt_len": tgt_out.size(0),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

_CATH_RE = re.compile(r"\|\s*([\d.]+)")

def _extract_cath_labels(chopping_star: str) -> list:
    """Return all CATH class labels found in a chopping_star string."""
    return _CATH_RE.findall(chopping_star)

def build_sample_weights(train_df, alpha: float = 0.2) -> torch.Tensor:
    """Compute per-sample WeightedRandomSampler weights.

    Strategy: weight = 1 / freq^alpha, where alpha controls smoothing strength.

        alpha=0.0  ->  uniform sampling (no rebalancing)
        alpha=0.1  ->  ~3x max oversampling of rarest vs most common class
        alpha=0.2  ->  ~8x max oversampling  (recommended default)
        alpha=0.5  ->  ~184x (old sqrt formula, too aggressive)

    Using the rarest CATH label in a chain to set the chain's weight means
    multi-domain chains containing a rare domain get boosted — the target
    behaviour for improving rare-class coverage.

    For samples with no parseable CATH label (unknown domains), assign the
    median weight so they are sampled at the average rate.
    """
    chopping_stars = train_df["chopping_star"].astype(str).tolist()

    # Count frequency of every CATH label across the whole training set.
    freq: dict = {}
    for cs in chopping_stars:
        for label in _extract_cath_labels(cs):
            freq[label] = freq.get(label, 0) + 1

    raw_weights = []
    for cs in chopping_stars:
        labels = _extract_cath_labels(cs)
        if labels:
            rarest_freq = min(freq[lbl] for lbl in labels)
            raw_weights.append(1.0 / (rarest_freq ** alpha))
        else:
            raw_weights.append(None)  # placeholder; filled below

    # Fill unknowns with median of the known weights.
    known = [w for w in raw_weights if w is not None]
    median_w = float(np.median(known)) if known else 1.0
    raw_weights = [w if w is not None else median_w for w in raw_weights]

    weights = torch.tensor(raw_weights, dtype=torch.float)
    return weights

def collate_fn(batch, src_pad_id, tgt_pad_id):
    src = torch.nn.utils.rnn.pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=src_pad_id)
    tgt_in = torch.nn.utils.rnn.pad_sequence([b["tgt_in"] for b in batch], batch_first=True, padding_value=tgt_pad_id)
    tgt_out = torch.nn.utils.rnn.pad_sequence([b["tgt_out"] for b in batch], batch_first=True, padding_value=tgt_pad_id)
    src_len = torch.tensor([b["src_len"] for b in batch], dtype=torch.long)
    tgt_len = torch.tensor([b["tgt_len"] for b in batch], dtype=torch.long)
    src_text = [b["src_text"] for b in batch]
    tgt_text = [b["tgt_text"] for b in batch]
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_len": src_len,
        "tgt_len": tgt_len,
        "src_text": src_text,
        "tgt_text": tgt_text,
    }
