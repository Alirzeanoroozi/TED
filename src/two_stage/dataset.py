"""Dataset + collate for the two-stage model.

Each row (``sequence``, ``chopping_star``) becomes:
    * the raw sequence string (fed to the frozen ESM2 backbone),
    * a per-residue assignment array (0 = linker, k = domain id),  -> Stage A,
    * per-domain CATH level-id targets (K, 4),                     -> Stage B.

Domains fully truncated away by ``max_len`` are dropped and the remaining ones
relabelled, so assignment ids and CATH targets stay aligned.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from two_stage.cath_vocab import CathVocab
from two_stage.targets import parse_chopping_star, residue_assignment


class TwoStageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cath_vocab: CathVocab, max_len: int = 1022):
        self.df = df.reset_index(drop=True)
        self.cath_vocab = cath_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def seq_len(self, idx: int) -> int:
        return min(len(str(self.df.iloc[idx]["sequence"]).strip()), self.max_len)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).strip()[: self.max_len]
        nres = len(seq)
        cs = str(row["chopping_star"])

        parsed = parse_chopping_star(cs)
        assign_full = residue_assignment(parsed.domains, nres)

        present = [k for k in range(1, len(parsed.domains) + 1) if np.any(assign_full == k)]
        assign = np.zeros(nres, dtype=np.int64)
        cath_ids = np.zeros((len(present), 4), dtype=np.int64)
        for new_id, old_id in enumerate(present, start=1):
            assign[assign_full == old_id] = new_id
            cath_ids[new_id - 1] = self.cath_vocab.encode(parsed.domains[old_id - 1].cath)

        return {
            "sequence": seq,
            "nres": nres,
            "assign": assign,            # (nres,)
            "cath_ids": cath_ids,        # (K, 4)
            "chopping_star": cs,
        }


def collate_fn(batch: List[dict]) -> dict:
    B = len(batch)
    Lmax = max(item["nres"] for item in batch)

    assign = torch.zeros(B, Lmax, dtype=torch.long)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    sequences: List[str] = []
    chopping_star: List[str] = []
    nres_list: List[int] = []

    domains: List[Tuple[int, torch.Tensor]] = []
    cath_targets: List[np.ndarray] = []

    for b, item in enumerate(batch):
        L = item["nres"]
        sequences.append(item["sequence"])
        chopping_star.append(item["chopping_star"])
        nres_list.append(L)
        a = torch.from_numpy(item["assign"])
        assign[b, :L] = a
        mask[b, :L] = True

        n_dom = int(item["cath_ids"].shape[0])
        for k in range(1, n_dom + 1):
            res_mask = torch.zeros(Lmax, dtype=torch.bool)
            res_mask[:L] = a == k
            if res_mask.any():
                domains.append((b, res_mask))
                cath_targets.append(item["cath_ids"][k - 1])

    if cath_targets:
        cath_target_tensor = torch.from_numpy(np.stack(cath_targets, axis=0)).long()
    else:
        cath_target_tensor = torch.zeros(0, 4, dtype=torch.long)

    return {
        "sequences": sequences,
        "assign": assign,
        "mask": mask,
        "domains": domains,                  # list of (sample_idx, res_mask (Lmax,) bool)
        "cath_targets": cath_target_tensor,  # (N, 4)  also used as teacher-forced parents
        "chopping_star": chopping_star,
        "nres": nres_list,
    }


class LengthBucketedSampler(Sampler):
    """Group similar-length chains into batches to cut padding waste (recipe 1.4).

    Sorts within randomly-shuffled mega-buckets, forms contiguous batches, then
    shuffles batch order each epoch. Approximate but cheap and dependency-free.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_multiplier: int = 50,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_multiplier
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches: List[List[int]] = []
        for i in range(0, len(idx), self.bucket_size):
            mega = idx[i : i + self.bucket_size]
            mega.sort(key=lambda j: self.lengths[j])
            for b in range(0, len(mega), self.batch_size):
                batches.append(mega[b : b + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
