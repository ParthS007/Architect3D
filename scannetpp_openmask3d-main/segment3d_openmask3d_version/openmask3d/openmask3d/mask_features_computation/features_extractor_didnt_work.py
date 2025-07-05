import clip
import numpy as np
import imageio
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
import PIL
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt
from PIL import Image

#<GROUP20>
#The goal was to make the feature extractor run on the GPU, however two main issues occurred: 
# (1) The GPU memory was just too small 
# (2) Distributing the process over the GPUs didn't work out as the splitting up introduced errors

class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: InstanceMasks3D, 
                 vis_threshold, 
                 indices,
                 device='cuda'):
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        
        # Multi-GPU setup
        if device == 'cuda' and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = torch.device('cuda:0')
            print(f"[INFO] Using {self.num_gpus} GPUs for parallel processing")
        else:
            self.num_gpus = 1
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"[INFO] Primary device: {self.device}")
        
        # Don't precompute the huge tensor - compute on demand
        self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_view()
        
    def get_visible_points_view(self):
        # Initialization
        vis_threshold = self.vis_threshold
        indices = self.indices
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path        
        resolution = (1440, 1920)
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)
        
        # Distribute views across GPUs
        views_per_gpu = len(indices) // self.num_gpus
        
        projected_points = np.zeros((len(indices), n_points, 2), dtype=np.int32)
        visible_points_view = np.zeros((len(indices), n_points), dtype=bool)
        
        print(f"[INFO] Computing the visible points in each view.")
        print(f"[INFO] How many views?", indices)
        print(f"[INFO] Processing {len(indices)} views across {self.num_gpus} GPUs")
        
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * views_per_gpu
            if gpu_id == self.num_gpus - 1:
                end_idx = len(indices)  # Last GPU handles remaining views
            else:
                end_idx = (gpu_id + 1) * views_per_gpu
            
            if start_idx >= len(indices):
                break
                
            device_id = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            print(f"[INFO] GPU {gpu_id} processing views {start_idx} to {end_idx-1}")
            
            # Move data to specific GPU
            poses_gpu = torch.tensor(poses[start_idx:end_idx], dtype=torch.float32, device=device_id)
            X_gpu = torch.tensor(X, dtype=torch.float32, device=device_id)
            intrinsic_gpu = torch.tensor(intrinsic, dtype=torch.float32, device=device_id)
            
            for i in range(start_idx, end_idx):
                local_i = i - start_idx
                idx = indices[i]
                
                # STEP 1: get the projected points
                projected_points_not_norm = torch.matmul(torch.matmul(intrinsic_gpu, poses_gpu[local_i]), X_gpu.T).T
                mask = (projected_points_not_norm[:, 2] != 0)
                
                if mask.sum() > 0:
                    valid_points = projected_points_not_norm[mask]
                    projected_coords = torch.stack([
                        valid_points[:, 0] / valid_points[:, 2],
                        valid_points[:, 1] / valid_points[:, 2]
                    ], dim=1).int()
                    projected_points[i][mask.cpu().numpy()] = projected_coords.cpu().numpy()
                
                # STEP 2: occlusions computation
                depth_path = os.path.join(depths_path, f"{idx:06d}" + '.png')
                sensor_depth = np.asarray(Image.fromarray(imageio.imread(depth_path)).resize((1920, 1440))) / depth_scale
                sensor_depth_gpu = torch.tensor(sensor_depth, dtype=torch.float32, device=device_id)

                projected_points_gpu = torch.tensor(projected_points[i], dtype=torch.int32, device=device_id)
                inside_mask = (projected_points_gpu[:,0] >= 0) & (projected_points_gpu[:,1] >= 0) & \
                             (projected_points_gpu[:,0] < width) & (projected_points_gpu[:,1] < height)
                
                point_depth = projected_points_not_norm[:,2]
                print(f"[DEBUG][View {i}] Points inside image: {inside_mask.sum()} / {n_points}")
                
                if inside_mask.sum() > 0:
                    inside_indices = torch.where(inside_mask)[0]
                    pi_inside = projected_points_gpu[inside_indices].T
                    point_depth_inside = point_depth[inside_indices]
                    
                    visibility_mask = torch.abs(sensor_depth_gpu[pi_inside[1], pi_inside[0]] - point_depth_inside) <= vis_threshold
                    print(f"[DEBUG][View {i}] Points passing visibility threshold: {visibility_mask.sum()} / {inside_mask.sum()}")
                    
                    inside_mask_updated = torch.zeros_like(inside_mask)
                    inside_mask_updated[inside_indices] = visibility_mask
                    visible_points_view[i] = inside_mask_updated.cpu().numpy()
                else:
                    visible_points_view[i] = inside_mask.cpu().numpy()
            
            # Clear GPU memory for this GPU
            torch.cuda.empty_cache()
                
        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        # Compute on-demand to avoid storing huge tensor
        visible_mask_points = self.get_visible_points_in_view_for_mask(view, mask)
        if visible_mask_points.sum() > 0:
            true_values = np.where(visible_mask_points)
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    def get_visible_points_in_view_for_mask(self, view, mask):
        """Compute visible points for a specific view and mask on-demand"""
        # Use the least loaded GPU
        gpu_id = view % self.num_gpus
        device_id = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        
        # Convert masks to tensor if needed - ensure bool dtype
        if isinstance(self.masks.masks, np.ndarray):
            mask_tensor = torch.tensor(self.masks.masks[:, mask], dtype=torch.bool, device=device_id)
        else:
            mask_tensor = self.masks.masks[:, mask].bool().to(device_id)
        
        visible_view_tensor = torch.tensor(self.visible_points_view[view], dtype=torch.bool, device=device_id)
        visible_mask_points = (mask_tensor & visible_view_tensor)
        
        if visible_mask_points.sum() > 0:
            proj_points = torch.tensor(self.projected_points[view], dtype=torch.int32, device=device_id)[visible_mask_points]
            
            # Create 2D mask
            visible_2d = torch.zeros((self.resolution[0], self.resolution[1]), dtype=torch.bool, device=device_id)
            if len(proj_points) > 0:
                visible_2d[proj_points[:,1], proj_points[:,0]] = True
            
            return visible_2d.cpu().numpy()
        else:
            return np.zeros((self.resolution[0], self.resolution[1]), dtype=bool)
    
    def get_top_k_indices_per_mask(self, k):
        """Compute top-k views for each mask without storing the full tensor"""
        num_masks = self.masks.num_masks
        num_views = len(self.indices)
        
        # Compute points per view per mask on-demand
        points_per_view_per_mask = np.zeros((num_views, num_masks))
        
        print(f"[INFO] Computing top-k views for {num_masks} masks across {num_views} views")
        
        for view in tqdm(range(num_views)):
            gpu_id = view % self.num_gpus
            device_id = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            
            # Process multiple masks on the same GPU to reduce data transfer
            visible_view_tensor = torch.tensor(self.visible_points_view[view], dtype=torch.bool, device=device_id)
            projected_points_tensor = torch.tensor(self.projected_points[view], dtype=torch.int32, device=device_id)
            
            if isinstance(self.masks.masks, np.ndarray):
                masks_tensor = torch.tensor(self.masks.masks, dtype=torch.bool, device=device_id)
            else:
                masks_tensor = self.masks.masks.bool().to(device_id)
            
            for mask in range(num_masks):
                visible_mask_points = (masks_tensor[:, mask] & visible_view_tensor)
                if visible_mask_points.sum() > 0:
                    proj_points = projected_points_tensor[visible_mask_points]
                    if len(proj_points) > 0:
                        # Create 2D mask and count points
                        visible_2d = torch.zeros((self.resolution[0], self.resolution[1]), dtype=torch.bool, device=device_id)
                        visible_2d[proj_points[:,1], proj_points[:,0]] = True
                        points_per_view_per_mask[view, mask] = visible_2d.sum().item()
            
            # Clear cache periodically
            if view % 10 == 0:
                torch.cuda.empty_cache()
        
        # Get top-k indices
        topk_indices_per_mask = np.argsort(-points_per_view_per_mask, axis=0)[:k,:].T
        return topk_indices_per_mask
    
class FeaturesExtractor:
    def __init__(self, 
                 camera, 
                 clip_model, 
                 images, 
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device='cuda'):
        self.camera = camera
        self.images = images
        
        # Multi-GPU setup
        if device == 'cuda' and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = torch.device('cuda:0')
        else:
            self.num_gpus = 1
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"[INFO] FeaturesExtractor using {self.num_gpus} GPUs, primary device: {self.device}")
        
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices, device)
        self.predictor_sam = initialize_sam_model(self.device, sam_model_type, sam_checkpoint)
        
        # Load CLIP models on multiple GPUs for parallel processing
        self.clip_models = []
        self.clip_preprocesses = []
        for gpu_id in range(self.num_gpus):
            device_id = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            clip_model, clip_preprocess = clip.load(clip_model, device_id)
            self.clip_models.append(clip_model)
            self.clip_preprocesses.append(clip_preprocess)
    
    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, 768))
        
        np_images = self.images.get_as_np_list()
        #print("np_images", np_images)
        
        for mask in tqdm(range(num_masks)): 
            gpu_id = mask % self.num_gpus  # Distribute masks across GPUs
            device_id = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            current_clip_model = self.clip_models[gpu_id]
            current_clip_preprocess = self.clip_preprocesses[gpu_id]
            
            images_crops = []
            
            print("topk_indices_per_mask", topk_indices_per_mask)
            for view_count, view in enumerate(topk_indices_per_mask[mask]):
                
                # Get original mask points coordinates in 2d images
                visible_mask_2d = self.point_projector.get_visible_points_in_view_for_mask(view, mask)
                point_coords = np.transpose(np.where(visible_mask_2d == True))
                
                print("point_coords.shape[0] > 0)", point_coords.shape[0] > 0)
                if (point_coords.shape[0] > 0):
                    self.predictor_sam.set_image(np_images[view])
                    
                    # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    
                    if save_crops:
                        plt.imsave(os.path.join(out_folder, f"crop{mask}_{view}_0_best_mask_w_sam.png"), best_mask)

                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask).to(device_id), level, multi_level_expansion_ratio)
                        # Convert coordinates back to CPU for PIL operations
                        #x1, y1, x2, y2 = x1().item(), y1.cpu().item(), x2.cpu().item(), y2.cpu().item()
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
                        if(save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                            
                        # Compute CLIP features
                        cropped_img_processed = current_clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)
            
            print("if(len(images_crops) > 0)", len(images_crops) > 0)                
            if(len(images_crops) > 0):
                image_input = torch.stack(images_crops).to(device_id)
                with torch.no_grad():
                    image_features = current_clip_model.encode_image(image_input).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                mask_clip[mask] = image_features.mean(dim=0).cpu().numpy()
            
            # Clear cache periodically
            if mask % 10 == 0:
                for gpu_id in range(self.num_gpus):
                    torch.cuda.empty_cache()
        
        # Final cleanup
        for gpu_id in range(self.num_gpus):
            torch.cuda.empty_cache()
                    
        return mask_clip