#!/bin/bash
#SBATCH --job-name=test_esmfold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/test_esmfold.out
#SBATCH --error=logs/test_esmfold.out

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python src/esm_fold.py