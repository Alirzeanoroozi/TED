#!/bin/bash
#SBATCH --job-name=rclone_ted_copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --partition=cosbi
#SBATCH --account=cosbi
#SBATCH --output=/scratch/esevinc22/TED/TED/logs/rclone_ted_copy-%J.out
#SBATCH --error=/scratch/esevinc22/TED/TED/logs/rclone_ted_copy-%J.out

set -euo pipefail

cd /scratch/esevinc22/TED/TED
mkdir -p logs data

rclone --version

# Only copy files that do not exist locally; never overwrite existing files.
rclone copy gdrive:TED_sequences /scratch/esevinc22/TED/TED/data --ignore-existing --fast-list --stats=30s
