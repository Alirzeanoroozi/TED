#!/bin/bash
#SBATCH --job-name=ted_diff_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/esevinc22/new_TED/logs/ted_diffusion_benchmark-%j.out
#SBATCH --error=/scratch/esevinc22/new_TED/logs/ted_diffusion_benchmark-%j.out

set -euo pipefail

REPO_DIR="/scratch/esevinc22/new_TED/TED"
SCRATCH_ROOT="/scratch/esevinc22/new_TED"
LOG_DIR="${SCRATCH_ROOT}/logs"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ted}"

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"

mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

echo "========== Slurm Job =========="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
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
print(f"Torch: {torch.__version__} cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="${CHECKPOINT:-artifacts/two_stage_diffusion_checkpoint.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/chainsaw_cath1363_from_scratch/results_two_stage_diffusion}"

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
    --deletion-fusion-alpha "${DELETION_FUSION_ALPHA:-1.0}" \
    --fusion-ablation \
    --boundary-window "${BOUNDARY_WINDOW:-2}" \
    --diagnostic-shrink-samples "${DIAGNOSTIC_SHRINK_SAMPLES:-8}" \
    --diagnostic-shrink-fraction "${DIAGNOSTIC_SHRINK_FRACTION:-0.10}"

echo "Benchmark finished at $(date)"
