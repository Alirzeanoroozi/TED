#!/bin/bash
#SBATCH --job-name=create_ted_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=cosbi
#SBATCH --account=cosbi
#SBATCH --output=/scratch/esevinc22/TED/TED/logs/create_ted_data-%J.out
#SBATCH --error=/scratch/esevinc22/TED/TED/logs/create_ted_data-%J.out

set -euo pipefail

cd /scratch/esevinc22/TED/TED
mkdir -p logs data/parquet_sequences

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" gmconda_py3923
export LD_LIBRARY_PATH="${CONDA_BASE}/lib:${LD_LIBRARY_PATH:-}"

python --version
python - <<'PY'
for module_name in ("pyarrow", "fastparquet", "polars"):
    try:
        __import__(module_name)
        print(f"{module_name}: OK")
    except Exception as exc:
        print(f"{module_name}: MISSING ({exc})")
PY
python src/data/create_data.py
