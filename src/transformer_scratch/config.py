from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "val_epochs": 1000,
        "lr": 10**-5,
        "src_vocab_size": 25,
        "tgt_vocab_size": 20,
        "src_seq_len": 2048,
        "tgt_seq_len": 256,
        "d_model": 960,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
    }

def get_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}_latest_new_new.pt"
    return str(Path('.') / model_folder / model_filename)
