import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
import random

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

from datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_200,
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200,
)



class ScannetPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "/work/courses/3dv/20/scannetpp", #<GROUP20>
        save_dir: str = "./work/scratch/dbagci/processed/scannetpp", #<GROUP20>
        modes: tuple = ("train", "validation"),
        n_jobs: int = 8,
        git_repo: str = "./data/raw/scannet/ScanNet",
        scannet200: bool = False,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        with open('/work/courses/3dv/20/scannetpp/metadata/instance_classes.txt', 'r', encoding="utf-8") as file: #<GROUP20>
            self.labels_pd = file.readlines()
        
        self.labels_pd = [x.strip() for x in self.labels_pd]

        self.create_label_database()
        for mode in self.modes:
            trainval_split_dir = '/work/courses/3dv/20/scannetpp/splits'
            scannet_special_mode = "val" if mode == "validation" else mode
            with open(
                f"/work/courses/3dv/20/scannetpp/splits/nvs_sem_{scannet_special_mode}.txt", encoding="utf-8"
            ) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]
            scans_folder = "scannetpp_ply"
            filepaths = []
            for scene in split_file:
                filepaths.append(
                    self.data_dir
                    / scans_folder
                    / scene
                    / "mesh_aligned_0.05.ply"
                )
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = {}
        for row_id, class_name in enumerate(self.labels_pd):
            label_database[row_id+1] = {
                "color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                "name": class_name,
                "validation": True,
                "row_id": row_id
            }
        self._save_yaml(
            self.save_dir / "label_database.yaml", label_database
        )
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        #scene, sub_scene = self._parse_scene_subscene(filepath.name)
        scene = filepath.parent.name
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))

        if mode in ["train", "validation"]:
            # getting scene information
            #description_filepath = Path(
            #    filepath
            #).parent / filepath.name.replace("_vh_clean_2.ply", ".txt")
            #with open(description_filepath) as f:
            #    scene_type = f.read().split("\n")[:-1]
            #scene_type = scene_type[-1].split(" = ")[1]
            #filebase["scene_type"] = scene_type
            #filebase["raw_description_filepath"] = description_filepath

            # getting instance info
            #<GROUP20>
            instance_info_filepath = next(
                Path(filepath).parent.glob("segments_anno.json")
            )
            segment_indexes_filepath = next(
                Path(filepath).parent.glob("segments.json")
            )
            instance_db = self._read_json(instance_info_filepath)
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            filebase["raw_instance_filepath"] = instance_info_filepath
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

            # reading labels file
            label_filepath = filepath.parent / filepath.name.replace(
                ".ply", "_semantic.ply"
            )
            filebase["raw_label_filepath"] = label_filepath
            label_coords, label_colors, labels = load_ply_with_normals(
                label_filepath
            )
            if not np.allclose(coords, label_coords):
                raise ValueError("files doesn't have same coordinates")

            # adding instance label
            labels = labels[:, np.newaxis]
            empty_instance_label = np.full(labels.shape, -1)
            labels = np.hstack((labels, empty_instance_label))
            for instance in instance_db["segGroups"]:
                #<GROUP20 START>
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"]

                label200 = instance["label"]
                #print("label200")
                #print(label200)
                #print("labels[occupied_indices, 0] before")
                #print(labels[occupied_indices, 0])
                #print("self.labels_pd")
                #print(self.labels_pd)
                # Map the category name to id
                try:
                    label_id = self.labels_pd.index(label200) + 1
                except ValueError:
                    label_id = -1  
                #label_id = self.labels_pd.index(label200)
                labels[occupied_indices, 0] = label_id
                #print("labels[occupied_indices, 0] after")
                #print(labels[occupied_indices, 0])
                #<GROUP20 END>
            points = np.hstack((points, labels))

            # gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1
            gt_data = points[:, -2] * 1000 + points[:, -1] + 1

        processed_filepath = (
            self.save_dir / mode / f"{scene}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode == "test":
            return filebase

        #<GROUP20>
        processed_gt_filepath = (
            self.save_dir
            / "instance_gt"
            / mode
            / f"{scene}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "/work/scratch/dbagci/processed/scannetpp/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

if __name__ == "__main__":
    Fire(ScannetPreprocessing)
