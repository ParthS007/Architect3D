#!/bin/bash

#SBATCH --account=3dv
#SBATCH --job-name=openmask3d
#SBATCH --output=./jobs/mask3d.o%j
#SBATCH --error=./jobs/mask3d.e%j
#SBATCH --time=8:59:59
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=8


export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# --- User Configuration: Adapt these paths for your ScanNet++ setup ---
# <GROUP20>
# 1. MASK_FILES_DIR: Directory containing your PRE-GENERATED ScanNet++ masks.
#    This is the output from your mask generation step.
#    The python script will look for mask files inside this directory.
#    Expected structure: Contains files like sceneXXXX_XX.npy (or whatever ctx.data.masks.masks_suffix matches)
# Verification: 'ls ${MASK_FILES_DIR}' should show your mask files (e.g., scene09c1414f1b.npy, etc. if you renamed them to fit the python script)
MASK_FILES_DIR="/work/courses/3dv/20/OpenArchitect3D/Mask3D/baseline_openmask3d_scannetpp_masks_generated_20250525_200612" #"/work/courses/3dv/20/OpenArchitect3D/my_scannetpp_masks_generated_20250519_094201" #"/work/courses/3dv/20/OpenArchitect3D/my_scannetpp_masks_generated_20250514_111913" # <<< YOUR PATH (e.g., output from previous mask generation step)
# /work/scratch/dbagci/processed/scannetpp/validation /work/courses/3dv/20/OpenArchitect3D/my_scannetpp_masks_generated_20250519_103729
# 2. MASK_FILES_SUFFIX: Glob pattern to find mask files within MASK_FILES_DIR.
#    The python script uses this suffix. For .npy files: "*.npy" or "scene*.npy"
# Verification: Ensure this pattern correctly matches your mask filenames.
MASK_FILES_SUFFIX="*.npy" # <<< Adjust if your mask files have a prefix like "scene"

# 3. RAW_SCANNETPP_DATA_ROOT: Path to the ROOT directory of your RAW ScanNet++ data.
#    The python script expects subfolders named 'sceneXXXX_XX' (or similar, based on parsed scene_num_str) inside this directory.
#    Each subfolder should contain: aligned_pose/, intrinsic/intrinsic_color.txt, color/, depth/, and the point cloud.
# Verification: 'ls ${RAW_SCANNETPP_DATA_ROOT}' should show your scene folders (e.g. scene09c1414f1b, or just 09c1414f1b - see critical note below)
RAW_SCANNETPP_DATA_ROOT="/work/courses/3dv/20/scannetpp/scannetpp_ply" # <<< YOUR PATH

# 4. OUTPUT_FEATURES_DIR: Directory where the computed features will be saved.
#    The python script will save one .npy feature file per scene here.
# Verification: Ensure you have write permissions.
OUTPUT_FEATURES_DIR="/work/courses/3dv/20/OpenArchitect3D/baseline_openmask3d_mask_features_scannetpp" # <<< YOUR PATH

# 5. SAM_CHECKPOINT_PATH: Path to your SAM checkpoint file.
# Verification: Ensure the .pth file exists.
#----------------------> CHANGE THIS
SAM_CHECKPOINT_PATH="/work/courses/3dv/20/sam_vit_h_4b8939.pth" # <<< YOUR PATH

# 6. PYTHON_SCRIPT_PATH: Full path to your dataset-level feature computation python script
#    (the second python script you provided in the last message).
PYTHON_SCRIPT_PATH="/work/courses/3dv/20/scannetpp_openmask3d-main/segment3d_openmask3d_version/openmask3d/openmask3d/compute_features_scannet200.py" #

# 7. HYDRA_CONFIG_NAME: The hydra config name used by the python script.
#    Your python script uses "openmask3d_scannet200_eval".
HYDRA_CONFIG_NAME="openmask3d_scannet200_eval"
#    The associated YAML file (e.g. openmask3d_scannet200_eval.yaml) must be in a 'configs'
#    directory relative to where the PYTHON_SCRIPT_PATH is, or relative to Original CWD if python script changes dir.
#    Your python script has: @hydra.main(config_path="configs", ...) and os.chdir(hydra.utils.get_original_cwd())
#    This means 'configs' should be relative to where you RUN this bash script.

# --- Paths RELATIVE to each scene folder within RAW_SCANNETPP_DATA_ROOT ---
# These are likely the same for ScanNet++ as for ScanNet200, but verify.
POSES_RELATIVE_PATH="aligned_pose"
INTRINSIC_RELATIVE_PATH="intrinsic/intrinsic_color.txt" # Path to file, not dir
IMAGES_RELATIVE_PATH="color"
DEPTHS_RELATIVE_PATH="depth"

# --- Other Parameters (match these to your ScanNet++ data and desired settings) ---
IMAGES_EXTENSION=".jpg"
DEPTHS_EXTENSION=".png"
INTRINSIC_RESOLUTION="[1440,1920]" # From your single-scene script
DEPTH_SCALE=1000                   # From your single-scene script
OPENMASK3D_FREQUENCY=3            # From your single-scene script
CLIP_MODEL_NAME="ViT-L/14@336px"         # Common default, verify from your Hydra config
SAM_MODEL_TYPE="vit_h"             # Common default, verify
OPTIMIZE_GPU_USAGE=true

# --- End of User Configuration ---

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_FEATURES_DIR}"

# Verify critical paths
if [ ! -d "${MASK_FILES_DIR}" ]; then echo "ERROR: MASK_FILES_DIR not found: ${MASK_FILES_DIR}"; exit 1; fi
if [ ! -d "${RAW_SCANNETPP_DATA_ROOT}" ]; then echo "ERROR: RAW_SCANNETPP_DATA_ROOT not found: ${RAW_SCANNETPP_DATA_ROOT}"; exit 1; fi
if [ ! -f "${SAM_CHECKPOINT_PATH}" ]; then echo "ERROR: SAM_CHECKPOINT_PATH not found: ${SAM_CHECKPOINT_PATH}"; exit 1; fi
if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then echo "ERROR: PYTHON_SCRIPT_PATH not found: ${PYTHON_SCRIPT_PATH}"; exit 1; fi

# Assuming this bash script is run from a directory that has a 'configs' subdirectory
# containing 'openmask3d_scannet200_eval.yaml' (and other configs it might include).
# The python script does os.chdir(hydra.utils.get_original_cwd()), so config_path="configs"
# will be relative to where this bash script is launched.

echo "Starting ScanNet++ feature computation for the entire dataset..."
echo "Reading masks from: ${MASK_FILES_DIR} (using suffix: ${MASK_FILES_SUFFIX})"
echo "Reading raw scan data from: ${RAW_SCANNETPP_DATA_ROOT}"
echo "Saving features to: ${OUTPUT_FEATURES_DIR}"

python "${PYTHON_SCRIPT_PATH}" \
    output.output_directory="${OUTPUT_FEATURES_DIR}" \
    data.masks.masks_path="${MASK_FILES_DIR}" \
    data.masks.masks_suffix="${MASK_FILES_SUFFIX}" \
    data.scans_path="${RAW_SCANNETPP_DATA_ROOT}" \
    data.camera.poses_path="${POSES_RELATIVE_PATH}" \
    data.camera.intrinsic_path="${INTRINSIC_RELATIVE_PATH}" \
    data.images.images_path="${IMAGES_RELATIVE_PATH}" \
    data.depths.depths_path="${DEPTHS_RELATIVE_PATH}" \
    data.images.images_ext="${IMAGES_EXTENSION}" \
    data.depths.depths_ext="${DEPTHS_EXTENSION}" \
    data.camera.intrinsic_resolution="${INTRINSIC_RESOLUTION}" \
    data.depths.depth_scale=${DEPTH_SCALE} \
    openmask3d.frequency=${OPENMASK3D_FREQUENCY} \
    external.sam_checkpoint="${SAM_CHECKPOINT_PATH}" \
    external.clip_model="${CLIP_MODEL_NAME}" \
    external.sam_model_type="${SAM_MODEL_TYPE}" \
    gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
    hydra.run.dir="${OUTPUT_FEATURES_DIR}/hydra_outputs" \
    #hydra.job.config.override_dirname.item_sep= "," \
    #    hydra.job.config.override_dirname.kv_sep= "=" \

    # Note: The hydra.job.config.override_dirname lines are to make hydra output paths cleaner if you have many overrides.
    # The HYDRA_CONFIG_NAME ("openmask3d_scannet200_eval") is passed by the @hydra.main decorator in the python script.

echo "Feature computation for ScanNet++ dataset finished."
echo "Features saved in: ${OUTPUT_FEATURES_DIR}"
## <GROUP20>