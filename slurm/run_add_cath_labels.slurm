#!/bin/bash
#SBATCH --job-name=add_cath_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --output=/scratch/esevinc22/new_TED/logs/add_cath_labels-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/add_cath_labels-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
LOG_DIR="${SCRATCH_ROOT}/logs"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

echo "========== Slurm Job =========="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "==============================="

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "conda command not found" >&2
    exit 1
fi

echo "Python: $(which python)"
python --version

export PYTHONUNBUFFERED=1

python benchmark/chainsaw_cath1363_from_scratch/add_cath_labels.py \
    --max_workers 8 \
    --retries 3 \
    --delay 0.3

echo "add_cath_labels finished at $(date)"
