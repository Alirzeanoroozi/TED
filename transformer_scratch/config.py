from pathlib import Path

def get_config():
    return {
        "batch_size": 6,
        "val_epochs": 500,
        "lr": 10**-4,
        "src_vocab_size": 25,
        "tgt_vocab_size": 20,
        "seq_len": 2048,
        "d_model": 960,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}_latest.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])