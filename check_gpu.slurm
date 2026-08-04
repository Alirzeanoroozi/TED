#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_gpu_%j.out

hostname
nvidia-smi --query-gpu=name --format=csv
nvidia-smi
