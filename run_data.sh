#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cosbi
#SBATCH --account=cosbi
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/comp.out
#SBATCH --error=logs/comp.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate new_bg
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python src/data/create_data.py