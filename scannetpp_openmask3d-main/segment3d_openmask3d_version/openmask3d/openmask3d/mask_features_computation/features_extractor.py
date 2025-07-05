
import clip
import numpy as np
import imageio
import torch
from tqdm import tqdm
import os
from data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
import PIL
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt
from PIL import Image

class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: InstanceMasks3D, 
                 vis_threshold, 
                 indices):
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        
    def get_visible_points_view(self):
        # Initialization
        vis_threshold = self.vis_threshold
        indices = self.indices
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path        
        resolution = (1440, 1920) #imageio.imread(os.path.join(depths_path, '0.png')).shape
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)
        
        projected_points = np.zeros((len(indices), n_points, 2), dtype = int)
        visible_points_view = np.zeros((len(indices), n_points), dtype = bool)
        print(f"[INFO] Computing the visible points in each view.")
        print(f"[INFO] How many views?", indices)
        for i, idx in tqdm(enumerate(indices)): # for each view
            # *******************************************************************************************************************
            # STEP 1: get the projected points
            # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
            projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
            # Get the mask of the points which have a non-null third coordinate to avoid division by zero
            mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
            # Get non homogeneous coordinates of valid points (2D in the image)
            projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
                    projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
            # *******************************************************************************************************************
            # STEP 2: occlusions computation
            # Load the depth from the sensor
            depth_path = os.path.join(depths_path, f"{idx:06d}" + '.png')
            sensor_depth = np.asarray(Image.fromarray(imageio.imread(depth_path)).resize((1920, 1440))) / depth_scale

            inside_mask = (projected_points[i,:,0] >= 0) * (projected_points[i,:,1] >= 0) \
                                * (projected_points[i,:,0] < width) \
                                * (projected_points[i,:,1] < height)
            pi = projected_points[i].T
            # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
            point_depth = projected_points_not_norm[:,2]
            #print(f"[DEBUG][View {i}] Points inside image: {inside_mask.sum()} / {n_points}")
            # Compute the visibility mask, true for all the points which are visible from the i-th view
            visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                        - point_depth[inside_mask]) <= \
                                        vis_threshold).astype(bool)
            #print(f"[DEBUG][View {i}] Points passing visibility threshold: {visibility_mask.sum()} / {inside_mask.sum()}")
            inside_mask[inside_mask == True] = visibility_mask
            visible_points_view[i] = inside_mask
        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    def get_visible_points_in_view_in_mask(self):
        masks = self.masks
        num_view = len(self.indices)
        visible_points_view, projected_points, resolution = self.get_visible_points_view()
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
        print(f"[INFO] Computing the visible points in each view in each mask.")
        for i in tqdm(range(num_view)):
            for j in range(masks.num_masks):
                visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
                print(f"[DEBUG][View {i}][Mask {j}] Visible mask points: {visible_masks_points.sum()}")
                proj_points = projected_points[i][visible_masks_points]
                if(len(proj_points) != 0):
                    print(f"[DEBUG][View {i}][Mask {j}] Projected points shape: {proj_points.shape}")
                    visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
                else:
                    print(f"[DEBUG][View {i}][Mask {j}] No projected points.")
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
        self.visible_points_view = visible_points_view
        self.projected_points = projected_points
        self.resolution = resolution
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution

    def get_visible_points_in_view_in_mask_new(self):
        masks = self.masks
        num_views = len(self.indices)
        visible_points_view, projected_points, resolution = self.get_visible_points_view()

        H, W = resolution
        num_masks = masks.num_masks
        num_points = self.point_cloud.num_points

        # Convert projected_points to torch (views x points x 2), and visibility mask
        projected_points_tensor = torch.from_numpy(projected_points).to(torch.int32).to("cuda")
        visible_points_view_tensor = torch.from_numpy(visible_points_view).to(torch.bool).to("cuda")
        #masks_tensor = torch.from_numpy(masks.masks).to(torch.bool).to("cuda")  # shape: (points, masks)

        # Preallocate output
        visible_points_in_view_in_mask = torch.zeros((num_views, num_masks, H, W), dtype=torch.bool, device="cuda")

        print(f"[INFO] Computing the visible points in each view in each mask (with GPU).")
        
        for view in tqdm(range(num_views)):
            visible_points = visible_points_view_tensor[view]  # shape: (points,)
            proj_pts = projected_points_tensor[view]  # shape: (points, 2)
            
            for mask_id in range(num_masks):
                # (points,) -> boolean mask for points that belong to this mask and are visible
                #visible_points_tensor = torch.from_numpy(visible_points).to(torch.bool).to("cuda")
                visible_points = visible_points.to("cuda:0")
                masks.masks = masks.masks.to("cuda:0")
                valid_mask = visible_points & masks.masks[:, mask_id]  # shape: (points,)
                if torch.any(valid_mask):
                    selected_pts = proj_pts[valid_mask]  # shape: (N, 2)
                    # Ensure all coords are within bounds
                    in_bounds = (selected_pts[:, 0] >= 0) & (selected_pts[:, 0] < W) & \
                                (selected_pts[:, 1] >= 0) & (selected_pts[:, 1] < H)
                    selected_pts = selected_pts[in_bounds]
                    if selected_pts.shape[0] > 0:
                        ys, xs = selected_pts[:, 1], selected_pts[:, 0]
                        visible_points_in_view_in_mask[view, mask_id, ys, xs] = True

        # Move back to numpy
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask.cpu().numpy()
        self.visible_points_view = visible_points_view_tensor.cpu().numpy()
        self.projected_points = projected_points_tensor.cpu().numpy()
        self.resolution = resolution
        return self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution

    
    def get_top_k_indices_per_mask(self, k):
        num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)
        print("num_points_in_view_in_mask", num_points_in_view_in_mask)
        topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
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
                 #rotation_deg_apply,
                 device):
        self.camera = camera
        self.images = images
        self.device = device
        #print("images", images)
        #print("images.indices",images.indices)
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
        #self.rotation_deg_apply = rotation_deg_apply
        #print("[INFO] Rotation correction for orientation set in FeaturesExtractor. Rotation to be applied:", self.rotation_deg_apply)
    
    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, 768)) #initialize mask clip
        
        np_images = self.images.get_as_np_list()
        #print("np_images", np_images)
        for mask in tqdm(range(num_masks)): # for each mask 
            images_crops = []
            if(optimize_gpu_usage):
                self.clip_model.to(torch.device('cpu'))
                self.predictor_sam.model.cuda()
            print("topk_indices_per_mask", topk_indices_per_mask)
            for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view
                if(optimize_gpu_usage):
                    torch.cuda.empty_cache()
                
                # Get original mask points coordinates in 2d images
                point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                print("point_coords.shape[0] > 0)", point_coords.shape[0] > 0)
                if (point_coords.shape[0] > 0):
                    self.predictor_sam.set_image(np_images[view])
                    
                    # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    
                    #no_sam_mask = self.point_projector.visible_points_in_view_in_mask[view][mask] == True
                    plt.imsave(os.path.join(out_folder, f"crop{mask}_{view}_0_best_mask_w_sam.png"), best_mask)
                    #plt.imsave(os.path.join(out_folder, f"crop{mask}_{view}_0_best_mask_no_sam.png"), no_sam_mask)

                    #pdb.set_trace()
                    #pdb.set_trace()
                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        

                        #if self.rotation_deg_apply !=0:
                        #    cropped_img = PIL.Image.fromarray(ndimage.rotate(cropped_img, self.rotation_deg_apply))
                            #print("[INFO]: Rotating crop by {} degrees".format(self.rotation_deg_apply))
                            #raise NotImplementedError
                        #else:
                            #print("[INFO]: Not rotating crop - rotation degree is {}".format(self.rotation_deg_apply))

                        if(save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                            
                        # I compute the CLIP feature using the standard clip model
                        cropped_img_processed = self.clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)
            
            if(optimize_gpu_usage):
                self.predictor_sam.model.cpu()
                self.clip_model.to(torch.device('cuda'))
            print("if(len(images_crops) > 0)", len(images_crops) > 0)                
            if(len(images_crops) > 0):
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True) #normalize
                
                mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()
                    
        return mask_clip
        
    