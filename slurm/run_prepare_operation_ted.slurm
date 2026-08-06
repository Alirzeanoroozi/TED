#!/bin/bash
#SBATCH --job-name=ted_prepare_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_prepare_v3-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_prepare_v3-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/all_parquet}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_DIR}/artifacts/operation_ted_v3/experiment}"
SPLIT_GROUP_COL="${SPLIT_GROUP_COL:-auto}"

mkdir -p "${SCRATCH_ROOT}/logs" "${EXPERIMENT_DIR}"
cd "${REPO_DIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "conda command not found" >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ "${REQUIRE_HOMOLOGY_GROUP:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--require_homology_group)
fi
if [[ "${OVERWRITE_EXPERIMENT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

python src/two_stage/prepare_experiment.py \
    --data_parquet_folder "${DATA_DIR}" \
    --manifest_path "${EXPERIMENT_DIR}/split_manifest.csv" \
    --cath_vocab_path "${EXPERIMENT_DIR}/cath_vocab.json" \
    --metadata_path "${EXPERIMENT_DIR}/metadata.json" \
    --split_group_col "${SPLIT_GROUP_COL}" \
    --val_ratio "${VAL_RATIO:-0.10}" \
    --test_ratio "${TEST_RATIO:-0.10}" \
    --split_seed "${SPLIT_SEED:-42}" \
    --cath_min_count "${CATH_MIN_COUNT:-5}" \
    "${EXTRA_ARGS[@]}"
