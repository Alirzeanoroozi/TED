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
source ${CONDA_BASE}/bin/activate latest_bg
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi

# python src/data/create_data.py
# python src/train_lightning.py
python src/evaluate.py
# python src/plot_att_weights.py