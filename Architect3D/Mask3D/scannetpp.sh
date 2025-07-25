#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=single_scene
#SBATCH --output=./jobs/1ada7a0617.o%j
#SBATCH --error=./jobs/1ada7a0617.e%j
#SBATCH --time=10:59:59
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
general.experiment_name="1ada7a0617" \
general.project_name="scannetpp_train" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode=train \
general.num_targets=2753 \
data.num_labels=2753 \
general.checkpoint="/work/courses/3dv/20/OpenArchitect3D/Mask3D/scannet200_val.ckpt" \





