#!/bin/bash
#SBATCH --job-name=ted_shrink_bench_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_shrink_bench_v3-%A_%a.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_shrink_bench_v3-%A_%a.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/artifacts/operation_ted_v3}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/benchmark/chainsaw_cath1363_from_scratch/results_operation_ted_v3}"
BENCHMARK_CSV="${BENCHMARK_CSV:-${REPO_DIR}/benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) RUN_ID=A_base_seed0 ;;
    1) RUN_ID=A_base_seed1 ;;
    2) RUN_ID=B_fixed_mask_seed0 ;;
    3) RUN_ID=B_fixed_mask_seed1 ;;
    4) RUN_ID=C_physical_aux_seed0 ;;
    5) RUN_ID=C_physical_aux_seed1 ;;
    6) RUN_ID=D_physical_integrated_seed0 ;;
    7) RUN_ID=D_physical_integrated_seed1 ;;
    *) echo "Bad SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2; exit 1 ;;
esac

CHECKPOINT="${EXPERIMENT_ROOT}/${RUN_ID}.pt"
OUTPUT_DIR="${RESULT_ROOT}/${RUN_ID}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${BENCHMARK_CSV}" ]]; then
    echo "Missing corrected benchmark: ${BENCHMARK_CSV}" >&2
    exit 1
fi

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p "${SCRATCH_ROOT}/logs" "${OUTPUT_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
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

python benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py \
    --checkpoint "${CHECKPOINT}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
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
