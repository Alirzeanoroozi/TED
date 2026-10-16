#!/bin/bash
#SBATCH --job-name=ted_domain_query_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/ted_domain_query_train.out
#SBATCH --error=logs/ted_domain_query_train.out

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python src/domain_query/train_lightning.py configs/domain_query_config.yaml
