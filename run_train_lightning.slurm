#!/bin/bash
#SBATCH --job-name=ted_lightning_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/esevinc22/TED/logs/ted_lightning_train-%j.out
#SBATCH --error=/scratch/esevinc22/TED/logs/ted_lightning_train-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/TED"
LOG_DIR="${SCRATCH_ROOT}/logs"
CHECKPOINT_DIR="${REPO_DIR}/lightning_logs"
ARTIFACT_DIR="${REPO_DIR}/artifacts"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/all_parquet}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${SCRATCH_ROOT}/.wandb_env}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${ARTIFACT_DIR}"

cd "${REPO_DIR}"

echo "========== Slurm Job =========="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR: ${DATA_DIR}"
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
print(f"Torch CUDA build: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found; continuing"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -f "${WANDB_ENV_FILE}" ]]; then
    # Load private W&B credentials from a non-repo file.
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi

export WANDB_PROJECT="${WANDB_PROJECT:-ted}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set. Create ${WANDB_ENV_FILE} with your W&B credentials." >&2
    exit 1
fi

TRAIN_CMD=(
    python src/train_lightning.py
    --data_parquet_folder "${DATA_DIR}"
    --save_dir "${CHECKPOINT_DIR}"
    --save_path "${ARTIFACT_DIR}/transformer_checkpoint.pt"
    --accelerator gpu
    --strategy auto
    --num_workers 4
    --benchmark_eval_steps 1000
    --benchmark_eval_subset_size 100
)

echo "Running command: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "Training finished at $(date)"
