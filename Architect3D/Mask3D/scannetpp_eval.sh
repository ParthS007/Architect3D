#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=mask3d
#SBATCH --output=./jobs/mask3d.o%j
#SBATCH --error=./jobs/mask3d.e%j
#SBATCH --time=0:07:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8

# Set PyTorch memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

export CURR_DBSCAN=0.95
export CURR_TOPK=300
export CURR_QUERY=150
export CURR_T=0.001


srun python main_instance_segmentation.py \
general.experiment_name="scannetpp_eval_class_agnostic_baseline" \
general.project_name="mask_computation" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
general.num_targets=2753 \
data.num_labels=2753 \
general.checkpoint="/work/courses/3dv/20/OpenArchitect3D/Mask3D/saved/final/last.ckpt" \
