#!/bin/bash
export OMP_NUM_THREADS=3 # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths

echo "In run ONE REPLICA ECCV - ${SCENE_NAME}"

# Paths are updated from your ScanNet++ configuration.
# The SCENE_NAME variable must be set before running this script (e.g., export SCENE_NAME=scene09c1414f1b)

# Base directory for a specific scene's raw data
SCENE_DIR="/work/courses/3dv/20/scannetpp/scannetpp_ply/${SCENE_NAME}"

# Relative paths from the first script are now appended to the SCENE_DIR
SCENE_POSE_DIR="${SCENE_DIR}/aligned_pose"
SCENE_COLOR_IMG_DIR="${SCENE_DIR}/color"
SCENE_DEPTH_IMG_DIR="${SCENE_DIR}/depth"

# Intrinsics file path for the scene
SCENE_INTRINSIC_PATH="${SCENE_DIR}/intrinsic/intrinsic_color.txt"

# --- NOTE: The PLY path was not explicitly in the first script's variables. ---
# --- You must verify that your PLY files are located directly within the RAW_SCANNETPP_DATA_ROOT directory ---
# --- and follow this naming convention. Adjust if necessary. ---
SCENE_PLY_PATH="/work/courses/3dv/20/scannetpp/scannetpp_ply/${SCENE_NAME}.ply" # <-- VERIFY THIS PATH AND NAMING

# Image and data settings from the first script
SCENE_INTRINSIC_RESOLUTION="[1440,1920]"
IMG_EXTENSION=".jpg"
DEPTH_EXTENSION=".png"
DEPTH_SCALE=1000

# Model checkpoint path
SAM_CKPT_PATH="/work/courses/3dv/20/sam_vit_h_4b8939.pth"

# Output directories
# The base directory for all output features
OUTPUT_DIRECTORY="/work/courses/3dv/20/OpenArchitect3D/mask_features_scannetpp_single_scene"
# A sub-directory for this specific scene's output
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${SCENE_NAME}"

# Path to the pre-generated mask file for this scene
# Assumes mask files are named according to the scene name
SCENE_MASK_PATH="/work/courses/3dv/20/OpenArchitect3D/my_scannetpp_masks_generated_20250519_103729/${SCENE_NAME}.npy" #<-- VERIFY MASK FILENAME

# Script settings
SAVE_VISUALIZATIONS=true
SAVE_CROPS=true
OPTIMIZE_GPU_USAGE=true

# --- The python script path from the first file. You may need to adjust this ---
# --- depending on which script you intend to run (single scene vs. dataset) ---
PYTHON_SCRIPT_PATH="/work/courses/3dv/20/scannetpp_openmask3d-main/segment3d_openmask3d_version/openmask3d/openmask3d/compute_features_single_scene.py"

# It seems the original script intended to run 'compute_features_single_scene_replica.py'.
# I will use that, but be aware of the PYTHON_SCRIPT_PATH defined above.
# If your single-scene script is in a different location, update the python command below.
cd /work/courses/3dv/20/scannetpp_openmask3d-main/segment3d_openmask3d_version/openmask3d/openmask3d/ #<-- Changed to script directory

echo "[INFO] Masks will be read from ${SCENE_MASK_PATH}."
echo "[INFO] Crops will be written to ${OUTPUT_FOLDER_DIRECTORY}/crops"

# 2. Compute mask features for each mask and save them
echo "[INFO] Computing mask features..."

# NOTE: The python script name is kept as in the original second file.
# You might need to change 'compute_features_single_scene_replica.py' to the correct script name.
python compute_features_single_scene_replica.py \
data.masks.masks_path=${SCENE_MASK_PATH} \
data.camera.poses_path=${SCENE_POSE_DIR} \
data.camera.intrinsic_path=${SCENE_INTRINSIC_PATH} \
data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
data.depths.depths_path=${SCENE_DEPTH_IMG_DIR} \
data.depths.depth_scale=${DEPTH_SCALE} \
data.images.images_path=${SCENE_COLOR_IMG_DIR} \
data.point_cloud_path=${SCENE_PLY_PATH} \
output.output_directory=${OUTPUT_FOLDER_DIRECTORY} \
output.save_crops=${SAVE_CROPS} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation" \
external.sam_checkpoint=${SAM_CKPT_PATH} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
openmask3d.frequency=1
#echo "[INFO] Feature computation done!"