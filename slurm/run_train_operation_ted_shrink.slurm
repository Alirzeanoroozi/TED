#!/bin/bash
#SBATCH --job-name=ted_shrink_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_shrink_v3-%A_%a.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_shrink_v3-%A_%a.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/all_parquet}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/artifacts/operation_ted_v3}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENT_ROOT}/experiment}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/lightning_logs_operation_ted_v3}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${SCRATCH_ROOT}/.wandb_env}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) ARM=A_base; SEED=0 ;;
    1) ARM=A_base; SEED=1 ;;
    2) ARM=B_fixed_mask; SEED=0 ;;
    3) ARM=B_fixed_mask; SEED=1 ;;
    4) ARM=C_physical_aux; SEED=0 ;;
    5) ARM=C_physical_aux; SEED=1 ;;
    6) ARM=D_physical_integrated; SEED=0 ;;
    7) ARM=D_physical_integrated; SEED=1 ;;
    *) echo "Bad SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2; exit 1 ;;
esac

DELETION_W=0.0
BUDGET_W=0.0
FIXED_MASK_W=0.0
PHYSICAL_W=0.0
BUDGET_SOURCE=supervised
INTEGRATION_ARGS=()
INIT_ARGS=()
case "${ARM}" in
    A_base) ;;
    B_fixed_mask)
        DELETION_W=0.25
        FIXED_MASK_W=0.25
        ;;
    C_physical_aux)
        DELETION_W=0.25
        PHYSICAL_W=0.25
        ;;
    D_physical_integrated)
        DELETION_W=0.25
        BUDGET_W=0.10
        PHYSICAL_W=0.25
        BUDGET_SOURCE=predicted
        INTEGRATION_ARGS+=(--deletion_feature_integration)
        ;;
esac

MANIFEST="${EXPERIMENT_DIR}/split_manifest.csv"
VOCAB="${EXPERIMENT_DIR}/cath_vocab.json"
if [[ ! -f "${MANIFEST}" || ! -f "${VOCAB}" ]]; then
    echo "Missing frozen experiment metadata. Run run_prepare_operation_ted.slurm first." >&2
    exit 1
fi

RUN_ID="${ARM}_seed${SEED}"
SAVE_DIR="${RUN_ROOT}/${RUN_ID}"
SAVE_PATH="${EXPERIMENT_ROOT}/${RUN_ID}.pt"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p "${SCRATCH_ROOT}/logs" "${SAVE_DIR}" "${EXPERIMENT_ROOT}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

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
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi
export WANDB_PROJECT="${WANDB_PROJECT:-ted_operation_shrink_v3}"
export WANDB_RUN_NAME="${RUN_ID}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

INIT_PATH="${INIT_FROM_CHECKPOINT:-${REPO_DIR}/artifacts/two_stage_checkpoint.pt}"
if [[ -f "${INIT_PATH}" ]]; then
    INIT_ARGS+=(--init_from_checkpoint "${INIT_PATH}")
else
    echo "Initialization checkpoint not found (${INIT_PATH}); training this arm from fresh heads."
fi

python src/two_stage/train.py \
    --data_parquet_folder "${DATA_DIR}" \
    --split_manifest_path "${MANIFEST}" \
    --cath_vocab_path "${VOCAB}" \
    --save_dir "${SAVE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --esm_model_name "${ESM_MODEL:-esm2_t30_150M_UR50D}" \
    --accelerator gpu \
    --strategy auto \
    --num_workers "${NUM_WORKERS:-4}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --epochs "${EPOCHS:-30}" \
    --max_steps "${MAX_STEPS:-30000}" \
    --limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}" \
    --shuffle_seed "${SPLIT_SEED:-42}" \
    --seed "${SEED}" \
    --save_every_n_train_steps "${SAVE_EVERY_N_TRAIN_STEPS:-2000}" \
    --max_len "${MAX_LEN:-1022}" \
    --lr "${LR:-1e-4}" \
    --warmup_steps "${WARMUP_STEPS:-1000}" \
    --residue_loss_weight 1.0 \
    --pair_loss_weight 1.0 \
    --deletion_loss_weight "${DELETION_W}" \
    --deletion_budget_loss_weight "${BUDGET_W}" \
    --linker_dropout_loss_weight "${FIXED_MASK_W}" \
    --empirical_shrink_loss_weight "${PHYSICAL_W}" \
    --empirical_shrink_max_deleted_fraction "${SHRINK_MAX_DELETED_FRACTION:-0.9}" \
    --empirical_shrink_full_state_probability "${SHRINK_FULL_STATE_PROBABILITY:-0.25}" \
    --artificial_deletion_loss_weight 0.0 \
    --deletion_head_input "${DELETION_HEAD_INPUT:-esm_trunk}" \
    --deletion_budget_source "${BUDGET_SOURCE}" \
    --cath_loss_type "${CATH_LOSS_TYPE:-cb_focal}" \
    --cb_beta "${CB_BETA:-0.999}" \
    --benchmark_eval_steps "${BENCHMARK_EVAL_STEPS:-2000}" \
    --benchmark_eval_subset_size "${BENCHMARK_EVAL_SUBSET_SIZE:-200}" \
    "${INTEGRATION_ARGS[@]}" \
    "${INIT_ARGS[@]}"

echo "Finished ${RUN_ID} at $(date)"
