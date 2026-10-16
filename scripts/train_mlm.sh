#!/bin/bash
#SBATCH --job-name=ted_residue_mlm_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/ted_residue_mlm_train.out
#SBATCH --error=logs/ted_residue_mlm_train.err

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# python src/residue_mlm/train_lightning.py configs/residue_mlm_config.yaml

sequence="MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"

python src/residue_mlm/predict_single.py --sequence $sequence --config configs/residue_mlm_config.yaml --checkpoint artifacts/residue_mlm_checkpoint.pt

