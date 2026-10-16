"""YAML config loader for residue-level MLM training."""

from __future__ import annotations

import argparse
from typing import Any

import yaml

REQUIRED_CONFIG_KEYS = {
    "data_parquet_folder",
    "batch_size",
    "max_len",
    "val_size",
    "test_size",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "random_crop_train",
    "mask_prob",
    "mask_aa",
    "mask_cath",
    "mask_domain_id",
    "lr",
    "weight_decay",
    "max_grad_norm",
    "max_epochs",
    "save_path",
    "save_dir",
    "checkpoint_filename",
    "checkpoint_monitor",
    "checkpoint_mode",
    "save_top_k",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "max_num_domains",
    "max_domain_index",
    "loss_weight_aa",
    "loss_weight_domain",
    "loss_weight_cath_c",
    "loss_weight_cath_a",
    "loss_weight_cath_t",
    "loss_weight_cath_h",
    "wandb_project",
    "wandb_run_name",
    "eval_sequence_prediction",
}


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {', '.join(missing)}")

    config["config"] = config_path
    return argparse.Namespace(**config)
