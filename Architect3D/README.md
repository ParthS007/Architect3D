# Architect3D: Enhanced 3D Instance Segmentation for Architectural Scenes

![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.3+-green)

## 🏗️ Overview

**Architect3D** is an advanced 3D instance segmentation model designed for architectural scene understanding. The project adapts the state-of-the-art [Mask3D](https://github.com/JonasSchult/Mask3D) model to work with the **ScanNet++** dataset, focusing on improving mask quality for complex architectural environments with 2,753 fine-grained classes.

### 🎯 Project Goals
- Improve mask quality of Mask3D for architectural scenes
- Adapt Mask3D to work with ScanNet++ dataset (2,753 classes)
- Integrate with OpenMask3D pipeline for open-vocabulary segmentation
- Provide comprehensive evaluation on architectural elements

### 🚨 Project Status & Limitations

> **Important**: Due to computational resource constraints (limited GPU hours, memory constraints, and 200GB storage limit per group), as well as time constraints and unexpected performance issues, we were unable to complete the full integration with the OpenMask3D pipeline. However, substantial work was done in preparing the codebase for this integration, and we include this preparatory work as it represents significant development effort.

## 📁 Repository Structure

```
Architect3D/
├── 📄 README.md                           # Project documentation
├── 📊 baseline.md                         # Evaluation baselines & results
├── 📋 requirements.txt                    # Python dependencies
├── 🎨 vis.py                             # t-SNE visualization generator
├── 🌐 interactive_tsne_visualization.html # Interactive class embeddings
├── 📑 Final_Report_Architect3D.pdf       # Comprehensive project report
│
├── 🏗️ Architect3D/                        # Main implementation
│   ├── Mask3D/                           # Adapted Mask3D for ScanNet++
│   │   ├── 🚀 main_instance_segmentation.py       # Main training/eval script
│   │   ├── ⚙️ conf/                                # Hydra configuration files
│   │   ├── 📊 benchmark/                           # Evaluation framework
│   │   │   └── evaluate_semantic_instance.py      # Core evaluation metrics
│   │   ├── 🗃️ datasets/                            # Dataset loaders & preprocessing
│   │   │   ├── preprocessing/                      # Data preprocessing scripts
│   │   │   │   └── scannetpp_preprocessing.py     # ScanNet++ specific preprocessing
│   │   │   ├── semseg.py                          # Dataset class implementation
│   │   │   └── utils.py                           # Dataset utilities
│   │   ├── 📈 jobs/                               # Training and evaluation logs
│   │   ├── 🧠 models/                             # Neural network architectures
│   │   ├── 🎯 trainer/                            # Training loops and logic
│   │   │   ├── __init__.py
│   │   │   └── trainer.py                         # Main training implementation
│   │   ├── 🔧 preprocessing.sh                    # ScanNet++ preprocessing script
│   │   ├── 🏃 scannetpp.sh                       # Training execution script
│   │   ├── 📊 scannetpp_eval.sh                  # Evaluation execution script
│   │   ├── 🛠️ utils/                              # General utilities
│   │   └── 💾 saved/final/                        # Model checkpoints & results
│   │       ├── visualizations/                    # Prediction visualizations
│   │       └── last.ckpt                          # Final model checkpoint
│   └── 📋 requirements.txt                        # Project dependencies
│
├── 🏠 scannetpp/                          # ScanNet++ dataset
│   ├── metadata/                          # Dataset metadata
│   │   ├── instance_classes.txt           # Class definitions
│   │   ├── scene_types.json              # Scene type mappings
│   │   ├── semantic_classes.txt          # Semantic class definitions
│   │   └── semantic_benchmark/           # Benchmark configurations
│   ├── scannetpp_ply/                    # 3D scene point clouds
│   │   ├── 0a5c013435/                   # Individual scene folders
│   │   └── 7b6477cb95/                   # (example scenes)
│   └── splits/                           # Dataset splits
│       ├── nvs_sem_train.txt             # Training split
│       ├── nvs_sem_val.txt               # Validation split
│       └── ...                           # Additional split files
│
├── 🔍 openmask3d/                         # OpenMask3D integration (prepared)
│   ├── 🏃 run_openmask3d_scannet200_eval.sh      # Evaluation script
│   ├── 🏃 run_openmask3d_single_scene.sh         # Single scene processing
│   └── openmask3d/                               # Core OpenMask3D modules
│       ├── 🎭 class_agnostic_mask_computation/   # Mask generation
│       ├── 🔮 mask_features_computation/         # Feature extraction
│       ├── 📊 evaluation/                        # Evaluation framework
│       └── 👁️ visualization/                     # Result visualization
│
├── 🏠 scannetpp_openmask3d-main/          # OpenMask3D adapted for ScanNet++
│   ├── scannetpp/                         # ScanNet++ specific code
│   └── segment3d_openmask3d_version/      # Segmentation pipeline
│
└── 📊 eval_results_architectural_classes/ # Evaluation results
    ├── 🎯 *.txt                          # Per-scene prediction results
    └── predicted_masks/                  # Generated instance masks
```

## 📊 Performance & Results

### Baseline Comparisons

| Model | Image Features | AP | AP50 | AP25 | Head (AP) | Common (AP) | Tail (AP) |
|-------|----------------|-----|------|------|-----------|-------------|-----------|
| **Closed-vocabulary (Supervised)** | | | | | | | |
| Mask3D | - | 26.9 | 36.2 | 41.4 | 39.8 | 21.7 | 17.9 |
| **Open-vocabulary** | | | | | | | |
| OpenMask3D-ScanNet200 | CLIP | 15.4 | 19.9 | 23.1 | 17.1 | 14.1 | 14.9 |
| **Architect3D-ScanNet++ (Ours)** | CLIP | *In Progress* | | | | | |

*Note: Complete evaluation results are pending due to computational constraints.*

### Key Achievements
- ✅ Successfully adapted Mask3D to ScanNet++ (2,753 classes)
- ✅ Implemented comprehensive evaluation framework
- ✅ Generated interactive t-SNE visualization of class embeddings
- ✅ Prepared OpenMask3D integration pipeline
- ✅ Optimized for architectural scene understanding

## 🚀 Quick Start

### Prerequisites
- **CUDA**: 11.3+
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **MinkowskiEngine**: For sparse 3D convolutions

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Architect3D

# Install dependencies
pip install -r requirements.txt

# For complete MinkowskiEngine setup, see detailed instructions below
```

## 🛠️ Usage Instructions

### Data Requirements
- **ScanNet++ Dataset**: Ensure access to the ScanNet++ dataset
- **File Structure**: Each scene folder should contain:
  - `mesh_aligned_0.05_semantic.ply`
  - `mesh_aligned_0.05.ply` 
  - `segments_anno.json`
  - `segments.json`

### Configuration Setup
Update paths in the following configuration files:
- [Main config file](Architect3D/Mask3D/conf/config_base_instance_segmentation.yaml)
- [Preprocessing script](Architect3D/Mask3D/preprocessing.sh)
- [Training script](Architect3D/Mask3D/scannetpp.sh)
- [Evaluation script](Architect3D/Mask3D/scannetpp_eval.sh)

> **Note**: This setup assumes usage of the [ETH Student Cluster](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster).

### Execution Steps

#### 1. Preprocess ScanNet++
```bash
cd Architect3D/Mask3D/
sbatch preprocessing.sh
```

#### 2. Run Model Evaluation
```bash
cd Architect3D/Mask3D/
sbatch scannetpp_eval.sh
```

#### 3. Generate Visualizations
```bash
# Create t-SNE visualization of class embeddings
python vis.py

# View interactive visualization
open interactive_tsne_visualization.html
```

### Codebase Adaptation and Challenges  

As mentioned above, we try to mark every written code with . This section serves as a overview of what we did with the existing codebase and what kind of challenges occurred.
We modified the official **Mask3D** repository—originally designed for **ScanNet**, **ScanNet200**, **S3DIS**, and **STPLS3D**—to support the **ScanNet++** dataset. Below, we outline the key adjustments and challenges encountered during this process.  

##### Environment Setup Issues  

The initial setup proved challenging due to dependencies, particularly the **MinkowskiEngine**, which is required for Mask3D. Known compatibility issues ([see official repository](https://github.com/JonasSchult/Mask3D#dependencies-memo)) were further complicated by unsupported **CUDA** versions on the student cluster. After consulting with supervisors and implementing workarounds, we successfully configured the environment. We include a workaround at the [end of the README](#workaround-for-minkowskiengine)

##### Key Code Adjustments for ScanNet++

To adapt Mask3D for ScanNet++, we had to perform modification which included, but were not limited to:  

- **Dataset preprocessing & loading**:  
  - Updated the instance class list  
  - Ensured consistent mapping of IDs to class names between ground truth and model predictions  
  - Adjusted config files and trainer for ScanNet++  
- **Model & training enhancements**:  
  - Adapted the model head to accommodate more classes  
  - Improved efficiency to support finer voxel sizes for better segmentation  
  - Updated evaluation and training code  
  - Conducted sanity checks (e.g., single/five-scene training; see [job logs](/Architect3D/Mask3D/jobs/))  
- **Integration with OpenMask3D pipeline**:  
  - Code for generating mask proposals and features with **Architect3D** on ScanNet++  

##### Challenges & Resolutions

- Code understanding: The Mask3D repository is extensive, and ScanNet++’s novelty required significant effort to understand and adapt the codebase.
- Resource limitations: Despite granted requests for additional computational resources, constraints persisted, affecting experimentation speed.
- Dataset accessibility:
  - Reproduction of results on ScanNet200 or other datasets was infeasible due to unavailability of the data.
- Version conflicts: The student cluster restricted older PyTorch/CUDA versions, requiring updates to newer dependencies.  
- Performance optimization: Modified the code to handle ScanNet++’s higher resolution.  

This adaptation enabled seamless integration of **Architect3D** while maintaining Mask3D’s core functionality.  

### List of libraries

A list of used libraries can be found in the [environment.yml](/Architect3D/Mask3D/environment.yml) file. If any issues arise during the setup of the environment, espacially the setup of the MinkowskiEngine, please consult the official [Mask3D repository](https://github.com/JonasSchult/Mask3D).

## 🎓 Detailed Installation Guide

### Prerequisites
- **Operating System**: Linux (tested on ETH Student Cluster)
- **CUDA**: 11.3+
- **Python**: 3.8+
- **GPU Memory**: 8GB+ recommended

### Complete MinkowskiEngine Setup for ETH Student Cluster

<details>
<summary><strong>Step-by-Step Installation Instructions</strong></summary>

```

STEP 1 - load modules

 module load gcc/8.2.0 python_gpu/3.8.5 r/4.0.2 git-lfs/2.3.0 2>/dev/null cmake/3.9.4 qt/5.10.0 boost/1.74.0 eth_proxy npm/6.14.9 open3d openblas cuda/11.3.1 cudnn/8.2.1.32
 
STEP 2 - create virtual environment (change ENVNAME)

 python -m venv ENVNAME
 
STEP 3 - activate virtual environment (change ENVNAME)

 source ~/ENVNAME/bin/activate
 
STEP 4 - install torch etc.(mask3d backbone requirements) using the following:

 pip install torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

 pip install ninja==1.10.2.3

 pip install pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2

 pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8

 pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps

 pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
 
STEP 5 - start minkowski installation, first clone the repo and open the local copy

 mkdir local_minkowski

 cd local_minkowski

 git clone git@github.com:NVIDIA/MinkowskiEngine.git

 cd MinkowskiEngine
 
STEP 5 - open the downloaded MinkowskiEngine folder, and edit setup.py based on the following: (I read an issue about how to install minkowski on euler https://github.com/NVIDIA/MinkowskiEngine/issues/447)
 
in setup.py, comment out the following section:

 # args with return value

 #CUDA_HOME, argv = _argparse("--cuda_home", argv, False)

 #BLAS="openblas"

 #BLAS_INCLUDE_DIRS = "/cluster/apps/gcc-8.2.0/openblas-0.3.20-qooku73wexw75ydcx6uivlkxtcw7fkqq/include"

 #BLAS_LIBRARY_DIRS =  "/cluster/apps/gcc-8.2.0/openblas-0.3.20-qooku73wexw75ydcx6uivlkxtcw7fkqq/lib"

 #MAX_COMPILATION_THREADS = 1
 
replace it with:

 # args with return value

 CUDA_HOME, argv = _argparse("--cuda_home", argv, False)

 BLAS="openblas"

 BLAS_INCLUDE_DIRS = "/cluster/apps/gcc-8.2.0/openblas-0.3.20-qooku73wexw75ydcx6uivlkxtcw7fkqq/include"

 BLAS_LIBRARY_DIRS =  "/cluster/apps/gcc-8.2.0/openblas-0.3.20-qooku73wexw75ydcx6uivlkxtcw7fkqq/lib"

 MAX_COMPILATION_THREADS = 1
 
STEP 6 - install minkowski

 python setup.py install
 
STEP 7 - check if minkowski is correctly installed

 [takmaza@eu-g3-084 MinkowskiEngine]$ python

 Python 3.8.5 (default, Sep 27 2021, 10:10:37) 

 [GCC 8.2.0] on linux

 Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import MinkowskiEngine
>>>
 
STEP 8 - install other dependencies

 pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1

 pip install gorilla-core==0.2.7.8

 pip install matplotlib==3.7.2

 pip install cython

 pip install pycocotools==2.0.6

 pip install h5py==3.7.0

 pip install transforms3d==0.4.1

 pip install open3d==0.13.0

 pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

 pip install torchmetrics==0.11.0

 pip install setuptools==68.0.0

 pip install fvcore==0.1.5.post20221221 

 pip install cloudpickle==2.1.0

 pip install Pillow==9.3.0

 pip install urllib3==1.26.16
 
cd openmask3d/class_agnostic_mask_computation/third_party/pointnet2 && pip install .

 cd ../../../..
 
pip install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 --no-deps

 pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588 --no-deps

 pip install ftfy==6.1.1

 pip install regex==2023.10.3
 pip install -e .

```

</details>

## 📊 Evaluation & Visualization

### Interactive Class Visualization
The repository includes an interactive t-SNE visualization of all 2,753 ScanNet++ architectural classes:
- **File**: `interactive_tsne_visualization.html`
- **Purpose**: Explore class relationships and embeddings
- **Usage**: Open in browser for interactive exploration

### Evaluation Results
Detailed evaluation results for architectural classes are stored in:
- `eval_results_architectural_classes/`: Per-scene prediction files
- `baseline.md`: Comprehensive baseline comparisons

## 🤝 Acknowledgments & References

### Core Technologies
- **[Mask3D](https://github.com/JonasSchult/Mask3D)**: Foundation 3D instance segmentation model
- **[OpenMask3D](https://github.com/OpenMask3D)**: Open-vocabulary 3D segmentation (unofficial version)
- **[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)**: Enhanced 3D scene understanding dataset
- **[CLIP](https://github.com/openai/CLIP)**: Vision-language representation learning
- **[SAM](https://github.com/facebookresearch/segment-anything)**: Segment Anything Model

### Development Context
This project was developed as part of the **3D Vision course at ETH Zurich**. Special thanks to our supervisors for providing the OpenMask3D codebase and guidance throughout the project.

### Citation
If you use this work in your research, please cite:

```bibtex
@article{architect3d2024,
  title={Architect3D: Enhanced 3D Instance Segmentation for Architectural Scenes},
  author={[Project Team]},
  journal={3D Vision Course Project},
  year={2024},
  institution={ETH Zurich}
}
```

## 📄 Additional Resources

- **📑 [Final Report](Final_Report_Architect3D.pdf)**: Comprehensive project documentation
- **📊 [Baseline Results](baseline.md)**: Detailed performance comparisons
- **🎨 [Class Visualization](interactive_tsne_visualization.html)**: Interactive t-SNE embeddings
- **🔧 [Configuration Files](Architect3D/Mask3D/conf/)**: Hydra configuration setup
- **📈 [Training Logs](Architect3D/Mask3D/jobs/)**: Detailed training and evaluation logs

---

<div align="center">
<sub>🏗️ Built with dedication for advancing 3D architectural scene understanding 🏗️</sub>
</div>

```
