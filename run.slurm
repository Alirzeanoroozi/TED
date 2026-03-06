#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/torch.out
#SBATCH --error=logs/torch.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate new_bg
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Optional: enable Weights & Biases (set WANDB=1 to log to wandb)
# export WANDB=1
# export WANDB_API_KEY=your_key_if_not_logged_in

nvidia-smi

# torchrun --nproc_per_node=2 train.py --data_csv data.csv
# torchrun --nproc_per_node=2 src/train.py --data_csv data/data.csv
# python src/train_lightning.py
python src/evaluate.py
# python src/plot_att_weights.py
# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 train.py --data_csv data.csv --use_wandb
# With wandb: torchrun --nproc_per_node=2 train.py --data_csv data.csv --use_wandb
# Or set WANDB=1 in the env block above
