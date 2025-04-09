#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --account=3dv
#SBATCH --output=mask3d.out

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=300
CURR_QUERY=150
CURR_T=0.001

# TRAIN python -m datasets.preprocessing.scannetpp_preprocessing preprocess --data_dir="/work/scratch/dbagci" --save_dir="/work/scratch/dbagci/processed/scannetpp" \
python main_instance_segmentation.py \
general.experiment_name="train" \
general.project_name="scannetpp_train" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode=train

