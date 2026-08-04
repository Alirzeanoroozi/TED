#!/bin/bash
#SBATCH --job-name=ted_diffusion_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_diffusion_train-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_diffusion_train-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
LOG_DIR="${SCRATCH_ROOT}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/lightning_logs_two_stage_diffusion}"
ARTIFACT_DIR="${REPO_DIR}/artifacts"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/all_parquet}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${SCRATCH_ROOT}/.wandb_env}"

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${ARTIFACT_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

echo "========== Slurm Job =========="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "DATA_DIR: ${DATA_DIR}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "TORCH_HOME: ${TORCH_HOME}"
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
print(f"Torch: {torch.__version__} CUDA build: {torch.version.cuda} available: {torch.cuda.is_available()}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_TRIM_THRESHOLD_=0

if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a; source "${WANDB_ENV_FILE}"; set +a
fi
export WANDB_PROJECT="${WANDB_PROJECT:-ted_diffusion}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-two_stage_diffusion_150M}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set. Create ${WANDB_ENV_FILE} with your W&B credentials." >&2
    exit 1
fi

RESUME_CKPT="${RESUME_CKPT:-}"
if [[ -f "${CHECKPOINT_DIR}/last.ckpt" ]]; then
    echo "Found ${CHECKPOINT_DIR}/last.ckpt; resuming."
    RESUME_CKPT="${CHECKPOINT_DIR}/last.ckpt"
fi

TRAIN_CMD=(
    python src/two_stage/train.py
    --data_parquet_folder "${DATA_DIR}"
    --save_dir "${CHECKPOINT_DIR}"
    --save_path "${ARTIFACT_DIR}/two_stage_diffusion_checkpoint.pt"
    --esm_model_name "${ESM_MODEL:-esm2_t30_150M_UR50D}"
    --accelerator gpu
    --strategy auto
    --num_workers "${NUM_WORKERS:-4}"
    --batch_size "${BATCH_SIZE:-4}"
    --epochs "${EPOCHS:-3}"
    --max_len "${MAX_LEN:-1022}"
    --lr "${LR:-1e-4}"
    --warmup_steps "${WARMUP:-1000}"
    --label_smoothing 0.1
    --residue_loss_weight "${RESIDUE_LOSS_WEIGHT:-1.0}"
    --pair_loss_weight "${PAIR_LOSS_WEIGHT:-1.0}"
    --deletion_loss_weight "${DELETION_LOSS_WEIGHT:-0.25}"
    --artificial_deletion_loss_weight "${ARTIFICIAL_DELETION_LOSS_WEIGHT:-0.10}"
    --artificial_noise_t_max "${ARTIFICIAL_NOISE_T_MAX:-0.9}"
    --artificial_noise_alpha_rate "${ARTIFICIAL_NOISE_ALPHA_RATE:-1.1}"
    --artificial_noise_max_insertions "${ARTIFICIAL_NOISE_MAX_INSERTIONS:-128}"
    --artificial_noise_max_len "${ARTIFICIAL_NOISE_MAX_LEN:-1022}"
    --cath_loss_type "${CATH_LOSS_TYPE:-cb_focal}"
    --cb_beta "${CB_BETA:-0.999}"
    --cath_curriculum_steps "${CATH_CURRICULUM_STEPS:-0}"
    --benchmark_eval_steps "${BENCHMARK_EVAL_STEPS:-2000}"
    --benchmark_eval_subset_size "${BENCHMARK_EVAL_SUBSET_SIZE:-100}"
    --benchmark_num_logged_samples 5
)

if [[ -n "${RESUME_CKPT}" ]]; then
    TRAIN_CMD+=(--resume_from_checkpoint "${RESUME_CKPT}")
else
    TRAIN_CMD+=(--init_from_checkpoint "${INIT_FROM_CHECKPOINT:-${ARTIFACT_DIR}/two_stage_checkpoint.pt}")
fi

echo "Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "Training finished at $(date)"
