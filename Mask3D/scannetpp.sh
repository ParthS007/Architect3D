#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=mask3d.out

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=300
CURR_QUERY=150
CURR_T=0.001

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="train3" \
general.project_name="scannetpp_train3" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode=train_validation

# TEST
python main_instance_segmentation.py \
general.experiment_name="scannetpp_benchmark_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_export_${CURR_T}_3" \
general.project_name="scannetpp_eval_3" \
data/datasets=scannetpp \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
general.export=true \
data.test_mode=test \
general.export_threshold=${CURR_T}
