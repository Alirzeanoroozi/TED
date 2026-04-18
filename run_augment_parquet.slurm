#!/bin/bash
#SBATCH --job-name=augment_parquet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=cosbi
#SBATCH --account=cosbi
#SBATCH --output=/scratch/esevinc22/TED/TED/logs/augment_parquet-%J.out
#SBATCH --error=/scratch/esevinc22/TED/TED/logs/augment_parquet-%J.out

set -euo pipefail

# Go to repo root first
cd /scratch/esevinc22/TED/TED
mkdir -p logs

# Initialize conda environment
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate gmconda_py3923
export LD_LIBRARY_PATH="${CONDA_BASE}/lib:${LD_LIBRARY_PATH:-}"

pwd
which python
python --version

python src/data/augment_parquet.py
