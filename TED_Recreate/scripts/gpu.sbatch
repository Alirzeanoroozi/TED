#!/bin/bash
#SBATCH -J ted
#SBATCH -A kutem
#SBATCH --qos=kutem
#SBATCH -p kutem_gpu
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
# Run a Python command of this repository on a GPU node. Submit from the repository root:
#   sbatch scripts/gpu.sbatch -m ted_recreate label --fasta proteins.fa --out labels.json
#   sbatch scripts/gpu.sbatch benchmark/03_run.py
# The account, partition and GPU type above are those of the KUTEM cluster; change them for yours.
set -euo pipefail
export TMPDIR=${TMPDIR:-$PWD/runs/tmp} PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
.venv/bin/python "$@"
