#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=mask3d
#SBATCH --output=./jobs/mask3d.o%j
#SBATCH --error=./jobs/mask3d.e%j
#SBATCH --time=2:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e # Exit on any error

# --- Configuration for Your ScanNet++ Mask Generation ---

# 1. MASK_MODULE_CKPT_PATH
MASK_MODULE_CKPT_PATH="/work/courses/3dv/20/scannetpp_openmask3d-main/segment3d_openmask3d_version/openmask3d/openmask3d/scannet200_val.ckpt" #"/work/courses/3dv/20/OpenArchitect3D/Mask3D/scannet200_val.ckpt"

# 2. SCANNETPP_PROCESSED_DATA_DIR
SCANNETPP_PROCESSED_DATA_DIR="/work/scratch/dbagci/processed/scannetpp"

# 3. SCANNETPP_LABEL_DB_PATH
SCANNETPP_LABEL_DB_PATH="${SCANNETPP_PROCESSED_DATA_DIR%/}/label_database.yaml"

# 4. MASK_OUTPUT_SAVE_DIR
#    Corrected date command for consistency if you want H_M_S here, or keep YYYYMMDD if preferred.
#    The original just had $(date +"%Y%m%d_%H%M%S")
#   "/work/scratch/dbagci/my_scannetpp_masks_generated_$(date +"%Y%m%d_%H%M%S")"
MASK_OUTPUT_SAVE_DIR="/work/courses/3dv/20/OpenArchitect3D/Mask3D/baseline_openmask3d_scannetpp_masks_generated_$(date +"%Y%m%d_%H%M%S")"

# 5. SAVE_VISUALIZATIONS
SAVE_VISUALIZATIONS=false

# 6. HYDRA_DATASET_CONFIG_NAME
HYDRA_DATASET_CONFIG_NAME="scannetpp"

# 7. TARGET_SPLIT_FOR_PROCESSING
TARGET_SPLIT_FOR_PROCESSING="validation"

# === FIX: Define a BASH variable for the experiment name string ===
BASH_GENERAL_EXPERIMENT_NAME="baseline_openmask3d_scannetpp_mask_gen_$(date +"%Y%m%d")"

# --- End of User Configuration ---

# Create the output directory
mkdir -p "${MASK_OUTPUT_SAVE_DIR}"
# Also create the base directory for hydra outputs if it's separate
mkdir -p "${MASK_OUTPUT_SAVE_DIR}/hydra_outputs/${BASH_GENERAL_EXPERIMENT_NAME}"


# Verify critical paths
if [ ! -f "${MASK_MODULE_CKPT_PATH}" ]; then echo "ERROR: Checkpoint not found: ${MASK_MODULE_CKPT_PATH}"; exit 1; fi
if [ ! -d "${SCANNETPP_PROCESSED_DATA_DIR}" ]; then echo "ERROR: Processed ScanNet++ data directory not found: ${SCANNETPP_PROCESSED_DATA_DIR}"; exit 1; fi
if [ ! -f "${SCANNETPP_LABEL_DB_PATH}" ]; then echo "ERROR: ScanNet++ label_database.yaml not found: ${SCANNETPP_LABEL_DB_PATH}"; exit 1; fi
# Assuming you run the script from OpenArchitect3D/Mask3D/
if [ ! -f "class_agnostic_mask_computation/conf/data/datasets/${HYDRA_DATASET_CONFIG_NAME}.yaml" ]; then echo "ERROR: Hydra dataset config 'conf/data/datasets/${HYDRA_DATASET_CONFIG_NAME}.yaml' not found. Create or verify name."; exit 1; fi

echo "--- Starting ScanNet++ Mask Generation ---"
echo "Current working directory: $(pwd)"
echo "Using Checkpoint: ${MASK_MODULE_CKPT_PATH}"
echo "Processed Data From: ${SCANNETPP_PROCESSED_DATA_DIR}"
echo "Label Database: ${SCANNETPP_LABEL_DB_PATH}"
echo "Saving Masks To: ${MASK_OUTPUT_SAVE_DIR}"
echo "Hydra Dataset Config: ${HYDRA_DATASET_CONFIG_NAME}.yaml"
echo "Targeting Split (via test_dataset config): ${TARGET_SPLIT_FOR_PROCESSING}"
echo "Hydra general.experiment_name will be: ${BASH_GENERAL_EXPERIMENT_NAME}"
echo "Hydra run directory will be: ${MASK_OUTPUT_SAVE_DIR}/hydra_outputs/${BASH_GENERAL_EXPERIMENT_NAME}"

python class_agnostic_mask_computation/get_masks_scannet200.py \
    general.experiment_name="${BASH_GENERAL_EXPERIMENT_NAME}" \
    general.project_name="OpenArchitect3D_on_ScanNetPP" \
    general.checkpoint="${MASK_MODULE_CKPT_PATH}" \
    general.train_mode=false \
    model.num_queries=100 \
    general.use_dbscan=true \
    general.dbscan_eps=0.95 \
    general.save_visualizations=${SAVE_VISUALIZATIONS} \
    data/datasets=${HYDRA_DATASET_CONFIG_NAME} \
    data.test_dataset.data_dir="${SCANNETPP_PROCESSED_DATA_DIR}"  \
    data.validation_dataset.data_dir="${SCANNETPP_PROCESSED_DATA_DIR}" \
    data.train_dataset.data_dir="${SCANNETPP_PROCESSED_DATA_DIR}" \
    data.test_dataset.label_db_filepath="${SCANNETPP_LABEL_DB_PATH}"  \
    data.validation_dataset.label_db_filepath="${SCANNETPP_LABEL_DB_PATH}" \
    data.train_dataset.label_db_filepath="${SCANNETPP_LABEL_DB_PATH}" \
    general.mask_save_dir="${MASK_OUTPUT_SAVE_DIR}" \
    hydra.run.dir="${MASK_OUTPUT_SAVE_DIR}/hydra_outputs/${BASH_GENERAL_EXPERIMENT_NAME}" # <<< USE THE BASH VARIABLE HERE
    #+data.test_dataset.split="${TARGET_SPLIT_FOR_PROCESSING}" \
    #hydra.run.dir="${MASK_OUTPUT_SAVE_DIR}/hydra_outputs/${BASH_GENERAL_EXPERIMENT_NAME}" # <<< USE THE BASH VARIABLE HERE

# Note on 'data.test_dataset.split': (This note remains the same)
# ...

echo "--- ScanNet++ Mask Generation Finished ---"
echo "Masks should be saved in: ${MASK_OUTPUT_SAVE_DIR}"
echo "Hydra logs should be in: ${MASK_OUTPUT_SAVE_DIR}/hydra_outputs/${BASH_GENERAL_EXPERIMENT_NAME}"