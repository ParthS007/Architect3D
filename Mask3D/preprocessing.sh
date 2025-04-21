#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=mask3d
#SBATCH --output=./jobs/mask3d.out
#SBATCH --error=./jobs/mask3d.err
#SBATCH --time=47:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8


srun python -m datasets.preprocessing.scannetpp_preprocessing preprocess --data_dir="/work/courses/3dv/20/scannetpp"  --save_dir="/work/scratch/habaumann/3dv/processed/scannetpp"
