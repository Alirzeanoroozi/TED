import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


def _normalize_target_column(df):
    """Use 'chopping_star' as target; parquet may use 'label'."""
    if "chopping_star" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "chopping_star"})
    return df


def _load_file(path):
    """Load a single file as DataFrame (parquet or csv)."""
    path = str(path)
    if path.lower().endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            if pl is None:
                raise ImportError(
                    "Reading parquet requires pandas parquet support (pyarrow/fastparquet) "
                    "or an installed polars fallback."
                ) from exc
            df = pd.DataFrame(pl.read_parquet(path).to_dicts())
    else:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    return _normalize_target_column(df)


def _load_paths(paths):
    """Load and concatenate file(s). paths: single path or list. Supports .parquet and .csv."""
    if isinstance(paths, str):
        paths = [paths]
    dfs = [_load_file(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["sequence", "chopping_star"])


def _load_csv_paths(paths):
    """Load and concatenate CSV(s). paths can be a single path or list of paths."""
    if isinstance(paths, str):
        paths = [paths]
    dfs = [pd.read_csv(p, encoding="utf-8", on_bad_lines="skip") for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = _normalize_target_column(df)
    return df.dropna(subset=["sequence", "chopping_star"])


class SequenceToChoppingDataset(Dataset):
    """Input: sequence (text), Target: chopping_star (text)."""

    def __init__(self, csv_paths, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=None):
        self.df = df if df is not None else _load_paths(csv_paths)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row["sequence"]).strip()[: self.max_src_len]
        tgt_text = str(row["chopping_star"]).strip()
        tgt_text_model = tgt_text[: self.max_tgt_len]
        tgt_was_truncated = len(tgt_text_model) != len(tgt_text)

        src_ids = self.src_tokenizer.encode(src_text, add_sos_eos=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text_model, add_sos_eos=True)

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
            "tgt_text_model": tgt_text_model,
            "tgt_was_truncated": tgt_was_truncated,
        }


def collate_fn(batch, pad_id_src, pad_id_tgt):
    src = torch.nn.utils.rnn.pad_sequence(
        [b["src"] for b in batch], batch_first=True, padding_value=pad_id_src
    )
    tgt_in = torch.nn.utils.rnn.pad_sequence(
        [b["tgt_in"] for b in batch], batch_first=True, padding_value=pad_id_tgt
    )
    tgt_out = torch.nn.utils.rnn.pad_sequence(
        [b["tgt_out"] for b in batch], batch_first=True, padding_value=pad_id_tgt
    )
    src_len = torch.tensor([b["src_len"] for b in batch], dtype=torch.long)
    tgt_len = torch.tensor([b["tgt_len"] for b in batch], dtype=torch.long)
    src_text = [b["src_text"] for b in batch]
    tgt_text = [b["tgt_text"] for b in batch]
    tgt_text_model = [b["tgt_text_model"] for b in batch]
    tgt_was_truncated = [b["tgt_was_truncated"] for b in batch]
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_len": src_len,
        "tgt_len": tgt_len,
        "src_text": src_text,
        "tgt_text": tgt_text,
        "tgt_text_model": tgt_text_model,
        "tgt_was_truncated": tgt_was_truncated,
    }


def create_train_val_datasets(
    csv_paths,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    max_tgt_len,
    val_ratio=0.1,
    shuffle_seed=42,
):
    """
    Create train and validation datasets from csv_paths.

    csv_paths: dict with "train" and optional "val" keys (each value is path or list of paths),
               or a single path / list of paths (will be split by val_ratio).
    Returns: (train_dataset, val_dataset)
    """
    if isinstance(csv_paths, dict):
        train_paths = csv_paths.get("train", [])
        val_paths = csv_paths.get("val", [])
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        if isinstance(val_paths, str):
            val_paths = [val_paths]
        train_df = _load_paths(train_paths)
        val_df = _load_paths(val_paths) if val_paths else None
    else:
        df = _load_paths(csv_paths)
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        n = len(df)
        n_val = max(1, int(n * val_ratio))
        train_df = df.iloc[: n - n_val]
        val_df = df.iloc[n - n_val :]

    val_df_final = val_df if val_df is not None and len(val_df) > 0 else train_df.iloc[:0]

    train_dataset = SequenceToChoppingDataset(
        [], src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=train_df
    )
    val_dataset = SequenceToChoppingDataset(
        [], src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=val_df_final
    )

    return train_dataset, val_dataset


def create_train_val_test_datasets(
    paths,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    max_tgt_len,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle_seed=42,
):
    """
    Create train, validation, and test datasets from a list of paths (parquet or csv).

    paths: list of file paths (or single path). Files are concatenated and split by ratios.
    Returns: (train_dataset, val_dataset, test_dataset)
    """
    df = _load_paths(paths)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = n
        n_val = 0
        n_test = 0
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val] if n_val else df.iloc[:0]
    test_df = df.iloc[n_train + n_val : n_train + n_val + n_test] if n_test else df.iloc[:0]

    train_dataset = SequenceToChoppingDataset(
        [], src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=train_df
    )
    val_dataset = SequenceToChoppingDataset(
        [], src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=val_df
    )
    test_dataset = SequenceToChoppingDataset(
        [], src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len, df=test_df
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(csv_paths, src_tok, tgt_tok, batch_size, max_src_len, max_tgt_len, world_size, rank, val_ratio=0.1):
    """Return (train_loader, val_loader) from csv_paths."""
    train_dataset, val_dataset = create_train_val_datasets(
        csv_paths, src_tok, tgt_tok, max_src_len, max_tgt_len, val_ratio=val_ratio
    )
    collate = lambda b: collate_fn(b, src_tok.pad_id, tgt_tok.pad_id)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader
