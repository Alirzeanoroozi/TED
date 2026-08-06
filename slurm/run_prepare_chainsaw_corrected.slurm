#!/bin/bash
#SBATCH --job-name=chainsaw_truth_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --output=/scratch/esevinc22/new_TED/logs/chainsaw_truth_fix-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/chainsaw_truth_fix-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"
BENCH_DIR="${REPO_DIR}/benchmark/chainsaw_cath1363_from_scratch"

mkdir -p "${SCRATCH_ROOT}/logs"
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

# Reconstruct truth from true_dbounds grouped by repeated true_dnames. Existing
# PDB files are reused; missing structures are downloaded by the preparation code.
python "${BENCH_DIR}/prepare_tedpred_benchmark.py" --write_parquet

# Rebuild labels in the same authoritative domain order. The checked-in CATH
# cache avoids repeating successful API lookups; unresolved entries may be fetched.
python "${BENCH_DIR}/add_cath_labels.py" \
    --max_workers "${CATH_API_WORKERS:-8}" \
    --retries "${CATH_API_RETRIES:-3}" \
    --delay "${CATH_API_DELAY:-0.3}"

python "${BENCH_DIR}/audit_cath_consistency.py" \
    --train-parquet-dir "${TRAIN_PARQUET_DIR:-${REPO_DIR}/data/all_parquet}" \
    --benchmark-csv "${BENCH_DIR}/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv" \
    --out "${BENCH_DIR}/cath_audit_report.json"

echo "Corrected CHAINSAW benchmark prepared at $(date)"
