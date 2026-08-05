#!/bin/bash
#SBATCH --job-name=ted_diff_ablate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_diff_ablate-%A_%a.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_diff_ablate-%A_%a.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
LOG_DIR="${SCRATCH_ROOT}/logs"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/all_parquet}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${SCRATCH_ROOT}/.wandb_env}"
ARTIFACT_ROOT="${REPO_DIR}/artifacts/ablation_task1"
CHECKPOINT_ROOT="${REPO_DIR}/lightning_logs_ablation_task1"

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_ROOT}" "${CHECKPOINT_ROOT}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) ARM=A; SEED=0; DELETION_W=0.0; ARTIFICIAL_W=0.0 ;;
    1) ARM=A; SEED=1; DELETION_W=0.0; ARTIFICIAL_W=0.0 ;;
    2) ARM=B; SEED=0; DELETION_W=0.25; ARTIFICIAL_W=0.0 ;;
    3) ARM=B; SEED=1; DELETION_W=0.25; ARTIFICIAL_W=0.0 ;;
    4) ARM=C; SEED=0; DELETION_W=0.25; ARTIFICIAL_W=0.10 ;;
    5) ARM=C; SEED=1; DELETION_W=0.25; ARTIFICIAL_W=0.10 ;;
    *) echo "Bad SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2; exit 1 ;;
esac

RUN_ID="ablation_${ARM}_seed${SEED}"
SAVE_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
SAVE_PATH="${ARTIFACT_ROOT}/${RUN_ID}.pt"
mkdir -p "${SAVE_DIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "conda command not found" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_TRIM_THRESHOLD_=0
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a; source "${WANDB_ENV_FILE}"; set +a
fi
export WANDB_PROJECT="${WANDB_PROJECT:-ted_diffusion_ablation}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set. Create ${WANDB_ENV_FILE} with your W&B credentials." >&2
    exit 1
fi

python src/two_stage/train.py \
    --data_parquet_folder "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --esm_model_name "${ESM_MODEL:-esm2_t30_150M_UR50D}" \
    --accelerator gpu \
    --strategy auto \
    --num_workers "${NUM_WORKERS:-4}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --epochs "${EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:-12000}" \
    --limit_val_batches "${LIMIT_VAL_BATCHES:-64}" \
    --val_ratio "${VAL_RATIO:-0.02}" \
    --test_ratio "${TEST_RATIO:-0.02}" \
    --shuffle_seed "${SHUFFLE_SEED:-42}" \
    --seed "${SEED}" \
    --save_every_n_train_steps "${SAVE_EVERY_N_TRAIN_STEPS:-1000}" \
    --max_len "${MAX_LEN:-1022}" \
    --lr "${LR:-1e-4}" \
    --warmup_steps "${WARMUP:-1000}" \
    --label_smoothing 0.1 \
    --residue_loss_weight "${RESIDUE_LOSS_WEIGHT:-1.0}" \
    --pair_loss_weight "${PAIR_LOSS_WEIGHT:-1.0}" \
    --deletion_loss_weight "${DELETION_W}" \
    --linker_dropout_loss_weight 0.0 \
    --artificial_deletion_loss_weight "${ARTIFICIAL_W}" \
    --deletion_head_input "${DELETION_HEAD_INPUT:-esm_trunk}" \
    --cath_loss_type "${CATH_LOSS_TYPE:-cb_focal}" \
    --cb_beta "${CB_BETA:-0.999}" \
    --cath_curriculum_steps "${CATH_CURRICULUM_STEPS:-0}" \
    --benchmark_eval_steps "${BENCHMARK_EVAL_STEPS:-2000}" \
    --benchmark_eval_subset_size "${BENCHMARK_EVAL_SUBSET_SIZE:-100}" \
    --benchmark_num_logged_samples 5 \
    --init_from_checkpoint "${INIT_FROM_CHECKPOINT:-${REPO_DIR}/artifacts/two_stage_checkpoint.pt}"

echo "Finished ${RUN_ID} at $(date)"
