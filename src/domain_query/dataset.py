"""Parquet-backed dataset for domain-query training: one residue mask + one CATH label per domain."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data._dataset import load_data, resolve_parquet_inputs
from residue_mlm.dataset import _target_column, split_dataframe
from residue_mlm.labels import (
    AA_TO_ID,
    PAD_ID,
    UNK_ID,
    LevelVocabulary,
    parse_domains_from_annotation,
    split_cath_levels,
)

CATH_LEVELS = ("cath_c", "cath_a", "cath_t", "cath_h")
IGNORE_ID = -100


def encode_domains(sequence: str, chopping_star: object, *, max_len: int, left_offset: int = 0, index_base: int = 1):
    """Crop the sequence and return (cropped_sequence, [(residue_indices, cath_label), ...])."""
    seq = str(sequence).strip().upper()[left_offset : left_offset + max_len]
    nres = len(seq)

    domains = []
    assigned: set[int] = set()
    for domain in parse_domains_from_annotation(chopping_star, index_base=index_base):
        residues: set[int] = set()
        for start, end in domain["segments"]:
            residues.update(range(max(0, start - left_offset), min(nres, end - left_offset)))
        residues = sorted(residues - assigned)
        if residues:
            domains.append((residues, domain["cath_label"]))
            assigned.update(residues)
    return seq, domains


def encode_cath_targets(cath_label: Optional[str], vocabs: Dict[str, LevelVocabulary]) -> List[int]:
    """No CATH label ("-") -> <unk> at every level, i.e. the "unassigned" class.
    A level missing below a known one (e.g. only "3.40") -> ignored in the loss."""
    if cath_label is None:
        return [UNK_ID] * len(CATH_LEVELS)
    levels = split_cath_levels(cath_label)
    return [
        vocabs[name].encode(value) if value is not None else IGNORE_ID
        for name, value in zip(CATH_LEVELS, levels)
    ]


def build_cath_vocabularies(df, target_column: str) -> Dict[str, LevelVocabulary]:
    vocabs = {name: LevelVocabulary(name) for name in CATH_LEVELS}
    for chopping_star in df[target_column]:
        for domain in parse_domains_from_annotation(chopping_star):
            for name, value in zip(CATH_LEVELS, split_cath_levels(domain["cath_label"])):
                vocabs[name].fit([value])
    return vocabs


class DomainQueryDataset(Dataset):
    def __init__(
        self,
        df,
        cath_vocabs: Dict[str, LevelVocabulary],
        *,
        max_len: int = 1022,
        random_crop: bool = False,
        target_column: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cath_vocabs = cath_vocabs
        self.max_len = max_len
        self.random_crop = random_crop
        self.target_column = target_column or _target_column(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        sequence = str(row["sequence"])
        chopping_star = row[self.target_column]

        left_offset = 0
        if self.random_crop and len(sequence) > self.max_len:
            left_offset = random.randint(0, len(sequence) - self.max_len)

        seq, domains = encode_domains(sequence, chopping_star, max_len=self.max_len, left_offset=left_offset)

        domain_masks = torch.zeros(len(domains), len(seq))
        for domain_idx, (residues, _) in enumerate(domains):
            domain_masks[domain_idx, residues] = 1.0
        domain_cath = torch.tensor(
            [encode_cath_targets(cath_label, self.cath_vocabs) for _, cath_label in domains],
            dtype=torch.long,
        ).reshape(-1, len(CATH_LEVELS))

        return {
            "sequence": seq,
            "aa_ids": torch.tensor([AA_TO_ID.get(aa, UNK_ID) for aa in seq], dtype=torch.long),
            "domain_masks": domain_masks,
            "domain_cath": domain_cath,
            "seq_len": len(seq),
            "left_offset": left_offset,
            "chopping_star": str(chopping_star),
        }


def collate_domain_batch(samples: List[Dict]) -> Dict:
    """Pad residues to the longest sequence in the batch; keep per-protein domain targets as lists."""
    max_len = max(sample["seq_len"] for sample in samples)
    aa_ids = torch.full((len(samples), max_len), PAD_ID, dtype=torch.long)
    for batch_idx, sample in enumerate(samples):
        aa_ids[batch_idx, : sample["seq_len"]] = sample["aa_ids"]

    return {
        "aa_ids": aa_ids,
        "seq_len": torch.tensor([sample["seq_len"] for sample in samples], dtype=torch.long),
        "sequences": [sample["sequence"] for sample in samples],
        "domain_masks": [sample["domain_masks"] for sample in samples],  # each: num_domains x seq_len
        "domain_cath": [sample["domain_cath"] for sample in samples],  # each: num_domains x 4
        "left_offsets": [sample["left_offset"] for sample in samples],
        "chopping_stars": [sample["chopping_star"] for sample in samples],
    }


def build_datasets(
    data_parquet_folder: str,
    *,
    max_len: int = 1022,
    val_size: int = 1000,
    test_size: int = 100,
    random_crop_train: bool = True,
):
    df = load_data(resolve_parquet_inputs(data_parquet_folder))
    target_column = _target_column(df)
    cath_vocabs = build_cath_vocabularies(df, target_column)
    train_df, val_df, test_df = split_dataframe(df, val_size=val_size, test_size=test_size)

    train_ds = DomainQueryDataset(train_df, cath_vocabs, max_len=max_len, random_crop=random_crop_train, target_column=target_column)
    val_ds = DomainQueryDataset(val_df, cath_vocabs, max_len=max_len, target_column=target_column)
    test_ds = DomainQueryDataset(test_df, cath_vocabs, max_len=max_len, target_column=target_column)
    return train_ds, val_ds, test_ds, cath_vocabs
