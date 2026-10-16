"""Parquet-backed dataset for residue-level MLM training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data._dataset import load_data, resolve_parquet_inputs
from residue_mlm.labels import (
    LevelVocabulary,
    MASK_ID,
    PAD_ID,
    UNK_ID,
    encode_residue_labels,
)


def _target_column(df) -> str:
    if "chopping_star" in df.columns:
        return "chopping_star"
    if "label" in df.columns:
        return "label"
    raise ValueError("Expected 'chopping_star' or 'label' column in parquet data.")


def build_level_vocabularies(df, target_column: str) -> Dict[str, LevelVocabulary]:
    vocabs = {
        "cath_c": LevelVocabulary("cath_c"),
        "cath_a": LevelVocabulary("cath_a"),
        "cath_t": LevelVocabulary("cath_t"),
        "cath_h": LevelVocabulary("cath_h"),
    }
    for _, row in df.iterrows():
        encoded = encode_residue_labels(
            row["sequence"],
            row[target_column],
            max_len=10_000,
            index_base=1,
            left_offset=0,
        )
        for level_name in vocabs:
            vocabs[level_name].fit(encoded[level_name])
    return vocabs


def split_dataframe(df, val_size: int = 1000, test_size: int = 100):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: n - val_size - test_size]
    val_df = df.iloc[n - val_size - test_size : n - test_size]
    test_df = df.iloc[n - test_size : n]
    return train_df, val_df, test_df


class ResidueMLMDataset(Dataset):
    def __init__(
        self,
        df,
        cath_vocabs: Dict[str, LevelVocabulary],
        *,
        max_len: int = 128,
        index_base: int = 1,
        random_crop: bool = False,
        target_column: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cath_vocabs = cath_vocabs
        self.max_len = max_len
        self.index_base = index_base
        self.random_crop = random_crop
        self.target_column = target_column or _target_column(df)

    def __len__(self) -> int:
        return len(self.df)

    def _choose_offset(self, seq_len: int) -> int:
        if seq_len <= self.max_len:
            return 0
        if self.random_crop:
            return random.randint(0, seq_len - self.max_len)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sequence = str(row["sequence"])
        chopping_star = row[self.target_column]
        left_offset = self._choose_offset(len(sequence))

        encoded = encode_residue_labels(
            sequence,
            chopping_star,
            max_len=self.max_len,
            index_base=self.index_base,
            left_offset=left_offset,
        )
        seq_len = encoded["seq_len"]

        cath_c = torch.tensor(
            [self.cath_vocabs["cath_c"].encode(v) for v in encoded["cath_c"]],
            dtype=torch.long,
        )
        cath_a = torch.tensor(
            [self.cath_vocabs["cath_a"].encode(v) for v in encoded["cath_a"]],
            dtype=torch.long,
        )
        cath_t = torch.tensor(
            [self.cath_vocabs["cath_t"].encode(v) for v in encoded["cath_t"]],
            dtype=torch.long,
        )
        cath_h = torch.tensor(
            [self.cath_vocabs["cath_h"].encode(v) for v in encoded["cath_h"]],
            dtype=torch.long,
        )

        return {
            "aa_ids": torch.tensor(encoded["aa_ids"], dtype=torch.long),
            "positions": torch.tensor(encoded["positions"], dtype=torch.long),
            "domain_ids": torch.tensor(encoded["domain_ids"], dtype=torch.long),
            "num_domains": torch.tensor(encoded["num_domains"], dtype=torch.long),
            "cath_c": cath_c,
            "cath_a": cath_a,
            "cath_t": cath_t,
            "cath_h": cath_h,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "left_offset": torch.tensor(encoded["left_offset"], dtype=torch.long),
            "chopping_star": str(chopping_star),
        }


def apply_mlm_mask(
    batch: Dict[str, torch.Tensor],
    *,
    mask_prob: float,
    aa_mask_id: int,
    level_mask_ids: Dict[str, int],
    mask_aa: bool = True,
    mask_cath: bool = True,
    mask_domain_id: bool = True,
) -> Dict[str, torch.Tensor]:
    """Apply BERT-style masking and return model inputs plus MLM targets."""
    device = batch["aa_ids"].device
    batch_size, max_len = batch["aa_ids"].shape
    seq_lens = batch["seq_len"]

    valid = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
    bernoulli = torch.rand(batch_size, max_len, device=device) < mask_prob
    mask_positions = valid & bernoulli

    outputs = dict(batch)
    outputs["mlm_mask"] = mask_positions

    if mask_aa:
        masked_aa = batch["aa_ids"].clone()
        replace = mask_positions & (torch.rand(batch_size, max_len, device=device) < 0.8)
        masked_aa[replace] = aa_mask_id
        random_replace = mask_positions & ~replace & (torch.rand(batch_size, max_len, device=device) < 0.5)
        if random_replace.any():
            from residue_mlm.labels import SPECIAL_VOCAB_SIZE, AA_ALPHABET

            random_tokens = torch.randint(
                low=SPECIAL_VOCAB_SIZE,
                high=SPECIAL_VOCAB_SIZE + len(AA_ALPHABET),
                size=masked_aa.shape,
                device=device,
            )
            masked_aa[random_replace] = random_tokens[random_replace]
        outputs["aa_ids_input"] = masked_aa
        outputs["aa_labels"] = batch["aa_ids"].clone()
        outputs["aa_labels"][~mask_positions] = -100
    else:
        outputs["aa_ids_input"] = batch["aa_ids"]
        outputs["aa_labels"] = torch.full_like(batch["aa_ids"], -100)

    for level_name, mask_id in level_mask_ids.items():
        key = level_name
        labels = batch[key].clone()
        if mask_cath:
            level_mask = mask_positions & (batch["domain_ids"] > 0)
            masked_level = batch[key].clone()
            masked_level[level_mask] = mask_id
            outputs[f"{key}_input"] = masked_level
            labels[~level_mask] = -100
        else:
            outputs[f"{key}_input"] = batch[key]
            labels[:] = -100
        outputs[f"{key}_labels"] = labels

    if mask_domain_id:
        domain_mask = mask_positions & (batch["domain_ids"] > 0)
        domain_input = batch["domain_ids"].clone()
        domain_input[domain_mask] = 0
        outputs["domain_ids_input"] = domain_input
        domain_labels = batch["domain_ids"].clone()
        domain_labels[~domain_mask] = -100
        outputs["domain_labels"] = domain_labels
    else:
        outputs["domain_ids_input"] = batch["domain_ids"]
        outputs["domain_labels"] = torch.full_like(batch["domain_ids"], -100)

    return outputs


def collate_residue_batch(samples: List[Dict], max_len: int = 128) -> Dict:
    batch_size = len(samples)
    out: Dict = {
        "aa_ids": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long),
        "positions": torch.zeros(batch_size, max_len, dtype=torch.long),
        "domain_ids": torch.zeros(batch_size, max_len, dtype=torch.long),
        "num_domains": torch.zeros(batch_size, dtype=torch.long),
        "cath_c": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long),
        "cath_a": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long),
        "cath_t": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long),
        "cath_h": torch.full((batch_size, max_len), PAD_ID, dtype=torch.long),
        "seq_len": torch.zeros(batch_size, dtype=torch.long),
    }

    for batch_idx, sample in enumerate(samples):
        length = int(sample["seq_len"])
        out["aa_ids"][batch_idx, :length] = sample["aa_ids"]
        out["positions"][batch_idx, :length] = sample["positions"]
        out["domain_ids"][batch_idx, :length] = sample["domain_ids"]
        out["num_domains"][batch_idx] = sample["num_domains"]
        out["cath_c"][batch_idx, :length] = sample["cath_c"]
        out["cath_a"][batch_idx, :length] = sample["cath_a"]
        out["cath_t"][batch_idx, :length] = sample["cath_t"]
        out["cath_h"][batch_idx, :length] = sample["cath_h"]
        out["seq_len"][batch_idx] = length

    out["left_offsets"] = [int(sample["left_offset"]) for sample in samples]
    out["chopping_stars"] = [sample["chopping_star"] for sample in samples]
    return out


def build_datasets(
    data_parquet_folder: str,
    *,
    max_len: int = 128,
    val_size: int = 1000,
    test_size: int = 100,
    random_crop_train: bool = True,
):
    parquet_files = resolve_parquet_inputs(data_parquet_folder)
    df = load_data(parquet_files)
    target_column = _target_column(df)
    cath_vocabs = build_level_vocabularies(df, target_column)
    train_df, val_df, test_df = split_dataframe(df, val_size=val_size, test_size=test_size)

    train_ds = ResidueMLMDataset(
        train_df,
        cath_vocabs,
        max_len=max_len,
        random_crop=random_crop_train,
        target_column=target_column,
    )
    val_ds = ResidueMLMDataset(
        val_df,
        cath_vocabs,
        max_len=max_len,
        random_crop=False,
        target_column=target_column,
    )
    test_ds = ResidueMLMDataset(
        test_df,
        cath_vocabs,
        max_len=max_len,
        random_crop=False,
        target_column=target_column,
    )
    return train_ds, val_ds, test_ds, cath_vocabs
