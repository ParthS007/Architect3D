from semantic.eval.eval_openmask3d import main

# test prev 400queries with 1.0 score - /cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS
'''
main(scene_list_file="/cluster/home/takmaza/scannetpp/nvs_sem_val.txt", 
     semantic_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/semantic_classes.txt", 
     instance_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/instance_classes.txt",  
     preds_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS",  
     gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",  
     data_root="/cluster/project/cvg/Shared_datasets/scannetpp/data",  
     check_pred_files=True)
'''

# test mask3d with 1.0 score - /cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_mask3d_masks_scannet200_PREDS
'''
main(scene_list_file="/cluster/home/takmaza/scannetpp/nvs_sem_val.txt", 
     semantic_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/semantic_classes.txt", 
     instance_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/instance_classes.txt",  
     preds_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_mask3d_masks_scannet200_PREDS",  
     gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",  
     data_root="/cluster/project/cvg/Shared_datasets/scannetpp/data",  
     check_pred_files=True)
'''

# test new 400queries_50 with 1.0 score - /cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50
'''
main(scene_list_file="/cluster/home/takmaza/scannetpp/nvs_sem_val.txt", 
     semantic_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/semantic_classes.txt", 
     instance_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/instance_classes.txt",  
     preds_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50",  
     gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",  
     data_root="/cluster/project/cvg/Shared_datasets/scannetpp/data",  
     check_pred_files=True)
#'''

# test new 400queries_50 with orig scores - /cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50_W_SCORES
'''
main(scene_list_file="/cluster/home/takmaza/scannetpp/nvs_sem_val.txt", 
     semantic_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/semantic_classes.txt", 
     instance_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/instance_classes.txt",  
     preds_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50_W_SCORES",  
     gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",  
     data_root="/cluster/project/cvg/Shared_datasets/scannetpp/data",  
     check_pred_files=True)
'''

# test gt
#main(scene_list_file="/cluster/home/takmaza/scannetpp/nvs_sem_val.txt", semantic_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/semantic_classes.txt", instance_classes_file="/cluster/project/cvg/Shared_datasets/scannetpp/metadata/semantic/instance_classes.txt",  preds_dir="/cluster/project/mtc/takmaza/scannetpp_val_gt_processed_NEW/gt_inst_pred_format",  gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",  data_root="/cluster/project/cvg/Shared_datasets/scannetpp/data",  check_pred_files=True)




# ECCV SCORES
#'''
#<GROUP20 START>
main(
     scene_list_file="/work/courses/3dv/20/scannetpp/splits/nvs_sem_val.txt", 
     semantic_classes_file="/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt", #"/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt", #"/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt", # /work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/metadata/instance_classes.txt
     instance_classes_file="/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt", #"/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt", #"/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt",#"/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt",  
     preds_dir="/work/courses/3dv/20/OpenArchitect3D/eval_results_architectural_classes",  
     gt_dir="/work/scratch/dbagci/processed/scannetpp/instance_gt/validation",  
     #gt_dir="/work/scratch/habaumann/3dv/processed/scannetpp/instance_gt/validation",  
     data_root="/work/courses/3dv/20/scannetpp/scannetpp_ply",  
     check_pred_files=True
)
#<GROUP20 END>
#'''
