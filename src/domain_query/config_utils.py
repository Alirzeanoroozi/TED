"""YAML config loader for domain-query training."""

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
    "esm_model_name",
    "freeze_esm",
    "num_encoder_layers",
    "d_model",
    "nhead",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
    "num_queries",
    "loss_weight_mask",
    "loss_weight_score",
    "loss_weight_cath",
    "score_threshold",
    "max_gap",
    "min_segment_len",
    "wandb_project",
    "wandb_run_name",
}


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {', '.join(missing)}")

    config["config"] = config_path
    return argparse.Namespace(**config)
