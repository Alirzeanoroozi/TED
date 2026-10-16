#!/bin/bash
#SBATCH -J ted_mmseqs_server
#SBATCH -A kutem
#SBATCH --qos=kutem
#SBATCH -p kutem_gpu
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH -o %x-%j.out
# Keep the TED lookup DB in memory and answer searches over HTTP (ted_recreate/server.py). Submit from the
# repository root:
#   sbatch scripts/mmseqs_server.sbatch                                          CPU: the index sits in RAM, no GPU
#   sbatch --gres=gpu:tesla_a100:1 --mem=64G scripts/mmseqs_server.sbatch gpu    GPU: the DB sits in GPU memory
# cath_label finds the server by itself. Stop it with scancel when you are done: it holds the RAM or the GPU.
set -euo pipefail
export TMPDIR=$PWD/db/tmp_server
mkdir -p "$TMPDIR"
exec .venv/bin/python -m ted_recreate.server --mode "${1:-cpu}"
