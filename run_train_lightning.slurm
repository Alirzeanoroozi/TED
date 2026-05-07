#!/bin/bash
#SBATCH --job-name=ted_lightning_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/ted_lightning_train-%j.out
#SBATCH --error=./logs/ted_lightning_train-%j.out

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

DATA_DIR="data/parquet_sequences"

python src/train_lightning.py 
