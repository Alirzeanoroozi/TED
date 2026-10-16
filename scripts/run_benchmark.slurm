#!/bin/bash
#SBATCH --job-name=ted_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_benchmark-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_benchmark-%j.out

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
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
python - <<'PY'
import torch
print(f"Torch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found; continuing"
fi

export PYTHONUNBUFFERED=1

python benchmark/chainsaw_cath1363_from_scratch/run_benchmark.py \
    --checkpoint artifacts/transformer_checkpoint_grammar.pt \
    --data-parquet-dir data/all_parquet \
    --device cuda \
    --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv \
    --output-dir benchmark/chainsaw_cath1363_from_scratch/results_with_CATH

echo "Benchmark finished at $(date)"
