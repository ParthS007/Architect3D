import yaml
import numpy as np

# Set these paths to your scene and label database
scene_files = ["/work/scratch/dbagci/processed/scannetpp/instance_gt/train/0a5c013435.txt", "/work/scratch/dbagci/processed/scannetpp/instance_gt/train/0c7962bd64.txt", "/work/scratch/dbagci/processed/scannetpp/instance_gt/train/0e350246d3.txt"]
label_db = "/work/scratch/dbagci/processed/scannetpp/label_database.yaml"

# Load label database
with open(label_db) as f:
    label_info = yaml.safe_load(f)

# Load ground truth instance ids
for scene_file in scene_files:
    ids = np.loadtxt(scene_file, dtype=int)
    label_ids = np.unique(ids // 1000)

    print("Label IDs in scene:", label_ids)
    print("Class names in scene:")
    for lid in label_ids:
        info = label_info.get(lid)
        if info is not None:
            print(f"{lid}: {info['name']}")
        else:
            print(f"{lid}: (not in label_database)")