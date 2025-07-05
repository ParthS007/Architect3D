#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=openmask3d_eval
#SBATCH --output=./jobs/openmask3d_eval.o%j
#SBATCH --error=./jobs/openmask3d_eval.e%j
#SBATCH --time=47:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
#set -e

# OPENMASK3D SCANNET200 EVALUATION SCRIPT
# This script performs the following in order to evaluate OpenMask3D predictions on the ScanNet200 validation set
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them
# 3. Evaluate for closed-set 3D semantic instance segmentation

# --------
# NOTE: SET THESE PARAMETERS!
SCANS_PATH="/work/courses/3dv/20/scannetpp/scannetpp_ply"
SCANNET_PROCESSED_DIR="/work/scratch/processed/scannetpp"
# model ckpt paths
MASK_MODULE_CKPT_PATH= "/work/courses/3dv/20/scannetpp_openmask3d-main/segment3d_openmask3d_version/openmask3d/openmask3d/mask3d_ckpt/last-epoch.ckpt" #"$(pwd)/resources/scannet200_val.ckpt"
SAM_CKPT_PATH=/work/courses/3dv/20/sam_vit_h_4b8939.pth #"$(pwd)/resources/sam_vit_h_4b8939.pth"
# output directories to save masks and mask features
EXPERIMENT_NAME="scannetpp_eval"
OUTPUT_DIRECTORY="$(pwd)/output"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
MASK_SAVE_DIR= "${OUTPUT_FOLDER_DIRECTORY}/masks" #"/work/courses/3dv/20/OpenArchitect3D/my_scannetpp_masks_generated_20250514_111913" #"${OUTPUT_FOLDER_DIRECTORY}/masks"
MASK_FEATURE_SAVE_DIR= "${OUTPUT_FOLDER_DIRECTORY}/mask_features" #"/work/courses/3dv/20/OpenArchitect3D/mask_features_scannetpp" #"${OUTPUT_FOLDER_DIRECTORY}/mask_features"
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations

# Paremeters below are AUTOMATICALLY set based on the parameters above:
SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"
# gpu optimization
OPTIMIZE_GPU_USAGE=false

#cd openmask3d

# 1.Compute class agnostic masks and save them
#python class_agnostic_mask_computation/get_masks_scannet200.py \
#general.experiment_name="scannetpp_eval" \
#general.project_name="scannetpp" \
#general.checkpoint=${MASK_MODULE_CKPT_PATH} \
#general.train_mode=false \
#model.num_queries=100 \
#general.use_dbscan=true \
#general.dbscan_eps=0.95 \
#general.save_visualizations=${SAVE_VISUALIZATIONS} \
#data.test_dataset.data_dir=${SCANNET_PROCESSED_DIR}  \
#data.validation_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
#data.train_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
#data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
#data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
#data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
#general.mask_save_dir=${MASK_SAVE_DIR} \
#echo "[INFO] Mask computation done!"
# get the path of the saved masks
#echo "[INFO] Masks saved to ${MASK_SAVE_DIR}."

#2. Compute mask features
#echo "[INFO] Computing mask features..."
#python compute_features_scannet200.py \
#data.scans_path=${SCANS_PATH} \
#data.masks.masks_path=${MASK_SAVE_DIR} \
#output.output_directory=${MASK_FEATURE_SAVE_DIR} \
#output.experiment_name=${EXPERIMENT_NAME} \
#external.sam_checkpoint=${SAM_CKPT_PATH} \
#gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
#hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
#echo "[INFO] Feature computation done!"

# 3. Evaluate for closed-set 3D semantic instance segmentation
python evaluation/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--mask_pred_dir=${MASK_SAVE_DIR} \
--mask_features_dir=${MASK_FEATURE_SAVE_DIR} \
