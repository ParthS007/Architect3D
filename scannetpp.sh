#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --account=3dv
#SBATCH --output=mask3d.out

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=300
CURR_QUERY=150
CURR_T=0.001

python -m datasets.preprocessing.scannetpp_preprocessing preprocess --data_dir="/work/courses/3dv/20/scannetpp" --save_dir="/work/courses/3dv/20/processed/scannetpp"


