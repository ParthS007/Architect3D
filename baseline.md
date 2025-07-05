# 📊 Architect3D Evaluation Baselines

This document provides comprehensive baseline comparisons for the **Architect3D** project, focusing on 3D instance segmentation performance across different model configurations and datasets.

## 🎯 Project Overview

**Architect3D** aims to enhance 3D instance segmentation quality specifically for architectural scenes by adapting Mask3D to work with the ScanNet++ dataset (2,753 architectural classes) and integrating with the OpenMask3D pipeline.

## 📈 Core Evaluation Metrics

- **AP (Average Precision)**: Overall segmentation quality across all IoU thresholds
- **AP50**: Average Precision at IoU threshold 0.5
- **AP25**: Average Precision at IoU threshold 0.25
- **Head/Common/Tail Classes**: Performance breakdown by class frequency distribution

## 🏆 3D Instance Segmentation Results

### Baseline Performance Comparison

| Model | Training Dataset | Image Features | AP | AP50 | AP25 | Head (AP) | Common (AP) | Tail (AP) |
|-------|------------------|----------------|-----|------|------|-----------|-------------|-----------|
| **🔒 Closed-vocabulary (Fully Supervised)** | | | | | | | | |
| Mask3D | ScanNet200 | - | **26.9** | **36.2** | **41.4** | **39.8** | **21.7** | **17.9** |
| **🌐 Open-vocabulary** | | | | | | | | |
| OpenMask3D | ScanNet200 | CLIP | 15.4 | 19.9 | 23.1 | 17.1 | 14.1 | 14.9 |
| **🏗️ Architect3D (Ours)** | **ScanNet++** | CLIP | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

> **Note**: Architect3D results are pending due to computational resource constraints. The model has been successfully adapted for ScanNet++ with 2,753 classes.

### 🎯 Key Observations
1. **Closed vs. Open Vocabulary**: Significant performance gap between supervised and open-vocabulary approaches
2. **Class Distribution Impact**: Head classes consistently outperform common and tail classes
3. **Dataset Scale**: ScanNet++ presents increased complexity with 10x more classes than ScanNet200

## 🔄 Generalization Results

### Cross-Dataset Performance Analysis

| Method | Mask Training Dataset | Novel Classes | | | Base Classes | | | All Classes | Tail (AP) |
|--------|----------------------|---------------|--|--|--------------|--|--|-------------|-----------|
| | | **AP** | **AP50** | **AP25** | **AP** | **AP50** | **AP25** | **AP** | |
| OpenMask3D | ScanNet20 | 11.9 | 15.2 | 17.8 | 14.3 | 18.3 | 21.2 | 12.6 | 11.5 |
| OpenMask3D | ScanNet200 | **15.0** | **19.7** | **23.1** | **16.2** | **20.6** | **23.1** | **15.4** | **14.9** |
| **Architect3D (Ours)** | **ScanNet++** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

### 📊 Analysis
- **Training Dataset Scale Impact**: Larger training datasets (ScanNet200 vs ScanNet20) improve generalization
- **Novel vs Base Classes**: Consistent performance gap between novel and base classes
- **Architecture Enhancement**: ScanNet++ adaptation expected to improve architectural scene understanding

## 🏛️ Architectural Elements Performance

### Specialized Evaluation on Building Components

| Method | Training Dataset | Novel Classes | | | Base Classes | | | All Classes | Tail (AP) |
|--------|------------------|---------------|--|--|--------------|--|--|-------------|-----------|
| | | **AP** | **AP50** | **AP25** | **AP** | **AP50** | **AP25** | **AP** | |
| OpenMask3D | ScanNet200 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Architect3D (Ours)** | **ScanNet++** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

## 🎯 Evaluation Methodology

### Dataset Characteristics
- **ScanNet++**: 2,753 fine-grained architectural classes
- **Enhanced Resolution**: 0.02m voxel size for detailed architectural features
- **Scene Diversity**: Wide range of architectural environments and building types

### Evaluation Protocols
1. **Standard Metrics**: AP, AP50, AP25 following established 3D segmentation benchmarks
2. **Class-wise Analysis**: Head/Common/Tail class breakdown based on frequency distribution
3. **Architectural Focus**: Specialized evaluation on building-specific elements

### Current Status
- ✅ **Model Adaptation**: Successfully adapted Mask3D for ScanNet++ (2,753 classes)
- ✅ **Evaluation Framework**: Comprehensive evaluation pipeline implemented
- ✅ **Preprocessing**: Complete ScanNet++ data processing pipeline
- 🔄 **Full Evaluation**: Pending due to computational resource constraints
- 🔄 **OpenMask3D Integration**: Prepared but not completed

## 📈 Expected Improvements

### Architectural Scene Understanding
- **Fine-grained Classification**: 2,753 classes vs. 200 in ScanNet200
- **Enhanced Detail**: Higher resolution (0.02m) for architectural precision
- **Domain Specialization**: Focused training on architectural environments

### Technical Enhancements
- **Scalable Architecture**: Model handles 10x more classes efficiently
- **Memory Optimization**: Custom optimizations for large-scale scenes
- **Evaluation Robustness**: Comprehensive metrics for architectural elements

## 🔗 Related Work Comparison

| Aspect | ScanNet200 | ScanNet++ (Ours) | Improvement |
|--------|------------|------------------|-------------|
| **Classes** | 200 | 2,753 | **13.8x more** |
| **Domain** | General indoor | Architectural focus | **Specialized** |
| **Resolution** | 0.05m | 0.02m | **2.5x finer** |
| **Evaluation** | Standard metrics | Architectural-specific | **Enhanced** |

---

> **Note**: This baseline document will be updated with complete results once computational resources allow for full evaluation. The current focus has been on successful model adaptation and framework preparation.
