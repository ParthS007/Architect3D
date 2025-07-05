#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=mask3d
#SBATCH --output=./jobs/mask3d.o%j
#SBATCH --error=./jobs/mask3d.e%j
#SBATCH --time=5:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8

python eval_open_vocab_100.py

python run_eval.py