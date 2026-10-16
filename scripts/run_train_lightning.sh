#!/bin/bash
#SBATCH --job-name=ted_lightning_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:7
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/comp.out
#SBATCH --error=logs/comp.out

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python src/train_lightning.py 
