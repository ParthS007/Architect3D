import os
import numpy as np
import clip
import torch
import pdb
#from eval_semantic_instance import evaluate
import tqdm
import argparse
from common.file_io import write_json
from common.utils.rle import rle_encode

label_list_100_path = "/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt"#top100_instance.txt" #"/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt"#architectural_instances.txt" #"/cluster/home/takmaza/scannetpp/LABEL_FILES/top100.txt"
with open(label_list_100_path, 'r') as f:
    LABEL_LIST_100 = [el.strip() for el in f.readlines()]
LABEL_TO_ID = {lbl:idx for idx, lbl in enumerate(LABEL_LIST_100)}
#LABEL_TO_ID = {'wall': 0, 'ceiling': 1, 'floor': 2, 'table': 3, 'door': 4, 'ceiling lamp': 5, 'cabinet': 6, 'blinds': 7, 'curtain': 8, 'chair': 9, 'storage cabinet': 10, 'office chair': 11, 'bookshelf': 12, 'whiteboard': 13, 'window': 14, 'box': 15, 'window frame': 16, 'monitor': 17, 'shelf': 18, 'doorframe': 19, 'pipe': 20, 'heater': 21, 'kitchen cabinet': 22, 'sofa': 23, 'windowsill': 24, 'bed': 25, 'shower wall': 26, 'trash can': 27, 'book': 28, 'plant': 29, 'blanket': 30, 'tv': 31, 'computer tower': 32, 'kitchen counter': 33, 'refrigerator': 34, 'jacket': 35, 'electrical duct': 36, 'sink': 37, 'bag': 38, 'picture': 39, 'pillow': 40, 'towel': 41, 'suitcase': 42, 'backpack': 43, 'crate': 44, 'keyboard': 45, 'rack': 46, 'toilet': 47, 'printer': 48, 'poster': 49, 'painting': 50, 'paper': 51, 'microwave': 52, 'board': 53, 'bottle': 54, 'bucket': 55, 'cushion': 56, 'power socket': 57, 'shoes': 58, 'basket': 59, 'shoe rack': 60, 'telephone': 61, 'file folder': 62, 'cloth': 63, 'blind rail': 64, 'laptop': 65, 'plant pot': 66, 'exhaust fan': 67, 'coat hanger': 68, 'light switch': 69, 'speaker': 70, 'table lamp': 71, 'papers': 72, 'air vent': 73, 'clothes hanger': 74, 'kettle': 75, 'shoe': 76, 'container': 77, 'power strip': 78, 'mug': 79, 'paper bag': 80, 'mouse': 81, 'smoke detector': 82, 'cup': 83, 'cutting board': 84, 'toilet paper': 85, 'paper towel': 86, 'pot': 87, 'slippers': 88, 'clock': 89, 'pan': 90, 'tap': 91, 'jar': 92, 'soap dispenser': 93, 'binder': 94, 'bowl': 95, 'tissue box': 96, 'whiteboard eraser': 97, 'socket': 98, 'toilet brush': 99}
#/cluster/home/takmaza/scannetpp/LABEL_FILES/top100.txt

INCLUDE_WALL_ETC = True

if INCLUDE_WALL_ETC:
    LABEL_LIST_100_INST = LABEL_LIST_100
    VALID_INST_LABELS = LABEL_LIST_100_INST
    VALID_INST_INDICES = set([LABEL_TO_ID[el] for el in VALID_INST_LABELS])

else:
    label_list_100inst_path = "/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/top100_instance.txt" #"/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt" #"/work/courses/3dv/20/scannetpp/metadata/semantic_benchmark/top100_instance.txt" #"/cluster/home/takmaza/scannetpp/LABEL_FILES/top100_instance.txt"
    with open(label_list_100inst_path, 'r') as f:
        LABEL_LIST_100_INST = [el.strip() for el in f.readlines()]
    VALID_INST_LABELS = LABEL_LIST_100_INST
    VALID_INST_INDICES = set([LABEL_TO_ID[el] for el in VALID_INST_LABELS]) #93 labels
#{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 
# 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 
# 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}
CLASS_LABELS = VALID_INST_LABELS

class InstSegEvaluator():
    def __init__(self, dataset_type, clip_model_type, sentence_structure):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] Device:", self.device)
        self.dataset_type = dataset_type
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.feature_size = self.get_feature_size(clip_model_type)
        print("[INFO] Feature size:", self.feature_size)
        print("[INFO] Getting label mapper...")
        self.set_label_and_color_mapper(dataset_type)
        print("[INFO] Got label mapper...")
        print("[INFO] Loading query sentences...")
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        print("[INFO] Loaded query sentences.")
        print("[INFO] Computing text query embeddings...")
        self.text_query_embeddings = self.get_text_query_embeddings().numpy() #torch.Size([20, 768])
        print("[INFO] Computed text query embeddings.")
        print("[INFO] Shape of query embeddings matrix:", self.text_query_embeddings.shape)

    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'scannetpp100':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_INST_INDICES)}.get)
            #self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_20.get)
        else:
            raise NotImplementedError

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):
        label_list = list(CLASS_LABELS)
        #label_list[-1] = 'other' # replace otherfurniture with other, following OpenScene
        return [sentence_structure.format(label) for label in label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            #print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings


    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, keep_first=None, scores_path=None):
        pred_masks = np.load(masks_path)
        mask_features = np.load(mask_features_path)

        keep_mask = np.asarray([True for el in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False

        # normalize mask features
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]
        mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

        per_class_similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(177, 20)
        print("## per_class_similarity_scores ##", mask_features)
        print(mask_features@self.text_query_embeddings.T)
        #print("## text_query_embeddings ##", self.text_query_embeddings)
        max_ind = np.argmax(per_class_similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)

        pred_masks = pred_masks[:, keep_mask]
        pred_classes = max_ind_remapped[keep_mask]

        if scores_path is not None:
            orig_scores = np.load(scores_path)
            pred_scores = orig_scores[keep_mask]
        else:
            pred_scores = np.ones(pred_classes.shape)

        return pred_masks, pred_classes, pred_scores

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_output.txt'):
        #pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score. 

        return inst_AP

def export_preds_in_scannet_format(pred_masks, pred_classes, pred_scores, pred_export_dir, scene_name):
    main_export_dir = pred_export_dir
    inst_masks_dir = os.path.join(main_export_dir, "predicted_masks")
    if not os.path.exists(inst_masks_dir):
        os.makedirs(inst_masks_dir)

    main_txt_file = os.path.join(main_export_dir, f'{scene_name}.txt')
    num_masks = pred_masks.shape[1]
    inst_ids = np.asarray(range(num_masks+1))
    inst_ids = inst_ids[inst_ids>0]
    main_txt_lines = []

    #pdb.set_trace()
    # for each instance
    for inst_ndx, inst_id in enumerate(sorted(inst_ids)):
        assert inst_ndx+1==inst_id
        # get the mask for the instance
        inst_mask = pred_masks[:, inst_ndx]
        # get the semantic label for the instance
        inst_sem_label = pred_classes[inst_ndx]
        # add a line to the main file with relative path
        # predicted_masks <semantic label> <confidence=1>
        mask_path_relative = f'predicted_masks/{scene_name}_{inst_ndx:03d}.json'
        inst_pred_score = pred_scores[inst_ndx]
        main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} {inst_pred_score}') #main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} 1.0')
        # save the instance mask to a file in the predicted_masks dir
        mask_path = os.path.join(main_export_dir, mask_path_relative)
        write_json(mask_path, rle_encode(inst_mask))    

    #pdb.set_trace()
    # save the main txt file
    with open(main_txt_file, 'w') as f:
        f.write('\n'.join(main_txt_lines))   
    """
                    # create main txt file
                main_txt_file = inst_predsformat_out_dir / f'{scene_id}.txt'
                # get the unique and valid instance IDs in inst_gt 
                # (ignore invalid IDs)
                inst_ids = np.unique(inst_gt)
                inst_ids = inst_ids[inst_ids > 0]
                # main txt file lines
                main_txt_lines = []

                # create the dir for the instance masks
                inst_masks_dir = inst_predsformat_out_dir / 'predicted_masks'
                inst_masks_dir.mkdir(parents=True, exist_ok=True)

                # for each instance
                for inst_ndx, inst_id in enumerate(tqdm(sorted(inst_ids))):
                # get the mask for the instance
                    inst_mask = inst_gt == inst_id
                    # get the semantic label for the instance
                    inst_sem_label = sem_gt[inst_mask][0]
                    # add a line to the main file with relative path
                    # predicted_masks <semantic label> <confidence=1>
                    mask_path_relative = f'predicted_masks/{scene_id}_{inst_ndx:03d}.json'
                    main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} 1.0')
                    # save the instance mask to a file in the predicted_masks dir
                    mask_path = inst_predsformat_out_dir / mask_path_relative
                    write_json(mask_path, rle_encode(inst_mask))

                # save the main txt file
                with open(main_txt_file, 'w') as f:
                    f.write('\n'.join(main_txt_lines))

    """

def test_pipeline_full_scannetpp100(mask_features_dir,
                                    gt_dir,
                                    pred_root_dir,
                                    sentence_structure,
                                    feature_file_template,
                                    pred_export_dir,
                                    dataset_type='scannetpp100',
                                    clip_model_type='ViT-L/14@336px',
                                    keep_first = None,
                                    scene_list_file='/work/courses/3dv/20/scannetpp/splits/nvs_sem_val.txt',
                                    masks_template='{}.npy',
                                    scores_dir=None,
                                    scores_template='{}.npy'
                                ):


    evaluator = InstSegEvaluator(dataset_type, clip_model_type, sentence_structure)
    print('[INFO]', dataset_type, clip_model_type, sentence_structure)

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()
    scene_names = sorted(scene_names)

    preds = {}

    if os.path.exists(pred_export_dir):
        print("Warning! Pred export dir already exists! - ", pred_export_dir)
        #raise Exception("Pred export dir already exists! - ", pred_export_dir)
    else:
        os.makedirs(pred_export_dir)

    for scene_name in tqdm.tqdm(scene_names[:]):
        masks_path = os.path.join(pred_root_dir, masks_template.format(scene_name))
        scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))
        if scores_dir is not None:
            scores_path = os.path.join(scores_dir, scores_template.format(scene_name))
        else:
            scores_path = None

        if not os.path.exists(scene_per_mask_feature_path):
            print('--- SKIPPING ---', scene_per_mask_feature_path)
            continue
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(masks_path=masks_path, 
                                                                                               mask_features_path=scene_per_mask_feature_path,
                                                                                               keep_first=keep_first,
                                                                                               scores_path=scores_path)
        
        export_preds_in_scannet_format(pred_masks, pred_classes, pred_scores, pred_export_dir, scene_name)

        #pdb.set_trace()
        #preds[scene_name] = {
        #    'pred_masks': pred_masks,
        #    'pred_scores': pred_scores,
        #    'pred_classes': pred_classes}

    #inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)


if __name__ == '__main__':

    # old 400queries with 1.0 score
    '''
    test_pipeline_full_scannetpp100(mask_features_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_data/w_segment3d_masks_400queries/features",
                                gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",
                                pred_root_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_data/w_segment3d_masks_400queries/masks",
                                sentence_structure="a {} in a scene",
                                feature_file_template='{}_openmask3d_features.npy',
                                dataset_type='scannetpp100',
                                clip_model_type='ViT-L/14@336px',
                                scene_list_file='/cluster/home/takmaza/scannetpp/nvs_sem_val.txt',
                                masks_template='{}.npy',
                                pred_export_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS"
                         )
    '''


    # mask3d preds with 1.0 score
    '''
    test_pipeline_full_scannetpp100(mask_features_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_data/w_mask3d_masks_scannet200/features",
                                gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",
                                pred_root_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_data/w_mask3d_masks_scannet200/masks",
                                sentence_structure="a {} in a scene",
                                feature_file_template='{}_openmask3d_features.npy',
                                dataset_type='scannetpp100',
                                clip_model_type='ViT-L/14@336px',
                                scene_list_file='/cluster/home/takmaza/scannetpp/nvs_sem_val.txt',
                                masks_template='{}.npy',
                                pred_export_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_mask3d_masks_scannet200_PREDS"
                         )
    '''


    # new 400queries_50 with 1.0 score
    '''
    test_pipeline_full_scannetpp100(mask_features_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_scannetpp/test_scannetpp_400queries_openmask3d_features_NEW/features",
                                gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",
                                pred_root_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_400queries_50/test_scannetpp_generate_masks_400queries_50/validation/masks",
                                sentence_structure="a {} in a scene",
                                feature_file_template='{}_openmask3d_features.npy',
                                dataset_type='scannetpp100',
                                clip_model_type='ViT-L/14@336px',
                                scene_list_file='/cluster/home/takmaza/scannetpp/nvs_sem_val.txt',
                                masks_template='{}.npy',
                                pred_export_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50"
                         )
    '''


    # new 400queries with orig_scores
    '''
    test_pipeline_full_scannetpp100(mask_features_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_scannetpp/test_scannetpp_400queries_openmask3d_features_NEW/features",
                                gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",
                                pred_root_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_400queries_50/test_scannetpp_generate_masks_400queries_50/validation/masks",
                                sentence_structure="a {} in a scene",
                                feature_file_template='{}_openmask3d_features.npy',
                                dataset_type='scannetpp100',
                                clip_model_type='ViT-L/14@336px',
                                scene_list_file='/cluster/home/takmaza/scannetpp/nvs_sem_val.txt',
                                masks_template='{}.npy',
                                pred_export_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/w_segment3d_masks_400queries_PREDS_50_W_SCORES",
                                scores_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_400queries_50/test_scannetpp_generate_masks_400queries_50/validation/scores"
                         )
    '''

    
    # ECCV SCORES
    '''
    test_pipeline_full_scannetpp100(mask_features_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/masks_segment3d_scannetpp_ECCV/features_exported",
                                gt_dir="/cluster/home/takmaza/scannetpp/GT_INST_100",
                                pred_root_dir="/cluster/scratch/takmaza/SEGMENT3D_ECCV/test_scannetpp_generate_masks/validation/masks/",
                                sentence_structure="a {} in a scene",
                                feature_file_template='{}_openmask3d_features.npy',
                                dataset_type='scannetpp100',
                                clip_model_type='ViT-L/14@336px',
                                scene_list_file='/cluster/home/takmaza/scannetpp/nvs_sem_val.txt',
                                masks_template='{}.npy',
                                pred_export_dir="/cluster/project/mtc/takmaza/scannetpp_masks_and_results/eval_exported_preds/segment3d_scannetpp_ECCV",
                         )
    '''

    test_pipeline_full_scannetpp100(    
        mask_features_dir="/work/courses/3dv/20/OpenArchitect3D/mask_features_scannetpp_single_scene",
        gt_dir="/work/scratch/dbagci/processed/scannetpp/instance_gt/validation",  # Update if needed  xxxxxxxxxxxxx
        pred_root_dir="/work/courses/3dv/20/OpenArchitect3D/Mask3D/masks",
        sentence_structure="a {} in a scene",  # Keep or adjust     
        feature_file_template="scene{}_openmask3d_features.npy",  # Adjust if file names differ
        dataset_type="scannetpp100",  # Update to "scannetpp_custom" or similar if needed
        clip_model_type="ViT-L/14@336px",  # Keep if features match
        scene_list_file="/work/courses/3dv/20/scannetpp/splits/nvs_sem_val.txt",  # Update if scenes differ xxxxxxxxxxxxxx
        masks_template="{}_masks.npy",  # Adjust if file names differ
        pred_export_dir="/work/courses/3dv/20/OpenArchitect3D/eval_results_architectural_classes"
    )