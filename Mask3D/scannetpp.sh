#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=mask3d
#SBATCH --output=./jobs/mask3d.out
#SBATCH --error=./jobs/mask3d.err
#SBATCH --time=47:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G


# Set PyTorch memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

export CURR_DBSCAN=0.95
export CURR_TOPK=300
export CURR_QUERY=150
export CURR_T=0.001


srun python main_instance_segmentation.py \
general.experiment_name="train" \
general.project_name="scannetpp_train" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode=train



