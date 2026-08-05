#!/bin/bash
#SBATCH --job-name=ted_diff_ab_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_diff_ab_bench-%A_%a.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_diff_ab_bench-%A_%a.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
LOG_DIR="${SCRATCH_ROOT}/logs"
ARTIFACT_ROOT="${REPO_DIR}/artifacts/ablation_task1"
RESULT_ROOT="${REPO_DIR}/benchmark/chainsaw_cath1363_from_scratch"

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) ARM=A; SEED=0 ;;
    1) ARM=A; SEED=1 ;;
    2) ARM=B; SEED=0 ;;
    3) ARM=B; SEED=1 ;;
    4) ARM=C; SEED=0 ;;
    5) ARM=C; SEED=1 ;;
    *) echo "Bad SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2; exit 1 ;;
esac

RUN_ID="ablation_${ARM}_seed${SEED}"
CHECKPOINT="${ARTIFACT_ROOT}/${RUN_ID}.pt"
OUTPUT_DIR="${RESULT_ROOT}/results_${RUN_ID}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

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

python benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py \
    --checkpoint "${CHECKPOINT}" \
    --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --batch-size "${BATCH_SIZE:-4}" \
    --min-domain-len "${MIN_DOMAIN_LEN:-20}" \
    --cluster-method "${CLUSTER_METHOD:-connected_components}" \
    --cath-decode "${CATH_DECODE:-beam}" \
    --cath-beam-width "${CATH_BEAM_WIDTH:-4}" \
    --max-len "${MAX_LEN:-1700}" \
    --deletion-fusion-alpha 0.0 \
    --boundary-window "${BOUNDARY_WINDOW:-2}" \
    --diagnostic-shrink-samples "${DIAGNOSTIC_SHRINK_SAMPLES:-0}"

echo "Finished benchmark ${RUN_ID} at $(date)"
