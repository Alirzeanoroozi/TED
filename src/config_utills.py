import argparse
import yaml

REQUIRED_CONFIG_KEYS = {
    "data_parquet_folder",
    "batch_size",
    "max_src_len",
    "max_tgt_len",
    "num_workers",
    "pin_memory",
    "persistent_workers",
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
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
    "benchmark_eval_steps",
    "weighted_sampling",
    "sampling_alpha",
    "grammar_guided_decoding",
    "wandb_project",
    "wandb_run_name",
}

def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {', '.join(missing)}")

    config["config"] = config_path
    return argparse.Namespace(**config)