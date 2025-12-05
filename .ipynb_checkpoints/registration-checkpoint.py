import argparse
import os
import numpy as np
import torch
import tifffile as tiff
import cv2
from skimage.filters import threshold_otsu
from skimage.transform import warp, ProjectiveTransform
from skimage import measure, morphology
from scipy import ndimage
from transformers import AutoImageProcessor, SuperGlueForKeypointMatching
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
import Global_deformation_functions as global_deform
import Local_deformation_512_v2_functions as local_deform
import divide_patch_functions as divide_patch
import pandas as pd
import math
import torch.nn.functional as F


def tissue_mask_by_chroma(img_bgr, blur=7, min_size=5000, close=11, otsu_bias=0):
    """
    Build a binary mask of tissue (1) vs background (0) using Lab chroma.

    - blur: Gaussian blur size to stabilize thresholding
    - min_size: remove tiny blobs (pixels)
    - close: morphological closing kernel size (pixels)
    - otsu_bias: add/subtract from the Otsu threshold (positive -> stricter mask)
    """
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a*a + b*b)  # colorfulness

    if blur > 0:
        k = int(blur) | 1
        chroma = cv2.GaussianBlur(chroma, (k, k), 0)

    # Otsu on uint8 version of chroma
    chroma_u8 = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    th_val, _ = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_val = np.clip(th_val + int(otsu_bias), 0, 255)

    mask = (chroma_u8 > th_val).astype(np.uint8)

    # Morphological cleanup
    if close > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    if min_size > 0:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size).astype(np.uint8)
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=min_size).astype(np.uint8)

    return mask  # uint8 {0,1}


def suppress_background(img_bgr, tissue_mask, bg_mode="dim", dim=0.15, bg_color=(245,245,245)):
    """
    Suppress background pixels where tissue_mask==0.

    bg_mode:
      - "dim": blend background toward its median color by (1-dim)
      - "fill": replace background with bg_color (e.g., white or gray)
      - "black": set to black
    """
    out = img_bgr.copy()

    bg_idx = (tissue_mask == 0)
    if not np.any(bg_idx):
        return out

    if bg_mode == "dim":
        # median background color
        med = np.median(out[bg_idx].reshape(-1, 3), axis=0)
        out[bg_idx] = (dim*out[bg_idx] + (1.0-dim)*med).astype(np.uint8)
    elif bg_mode == "fill":
        out[bg_idx] = np.array(bg_color, dtype=np.uint8)
    elif bg_mode == "black":
        out[bg_idx] = 0

    return out

def load_models(global_model_path, local_model_path):
    global_model = global_deform.DualEncoderUNet(in_ch=1)
    # checkpoint = torch.load("/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_real_sample_v3.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)
    checkpoint = torch.load(global_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    
    local_model = local_deform.DualEncoderUNet(in_ch=1)

    # checkpoint = torch.load("/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_real_local_512_v2_more2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)
    checkpoint = torch.load(local_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)

    local_model.load_state_dict(checkpoint['model_state_dict'])

    return global_model, local_model

def _sample_flow_at_points(flow_dy_dx, pts_xy):
    """
    flow_dy_dx: (2, H, W) tensor with channels (dy, dx) in **pixel units**
    pts_xy: (N, 2) tensor of (x, y) in **pixel coords** (same image size as flow)
    returns: (N, 2) sampled (dy, dx) at those points (bilinear)
    """
    device = flow_dy_dx.device
    _, H, W = flow_dy_dx.shape
    # normalize pts to [-1, 1] for grid_sample (x then y)
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    gx = 2.0 * x / (W - 1) - 1.0
    gy = 2.0 * y / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2)  # (1, N, 1, 2)

    flow = flow_dy_dx.unsqueeze(0)  # (1, 2, H, W)
    # Sample dy and dx
    sampled = F.grid_sample(flow, grid, mode='bilinear', align_corners=True)  # (1, 2, N, 1)
    sampled = sampled.squeeze(0).squeeze(-1).t()  # (N, 2) but order (dy, dx)
    dy = sampled[:, 0]
    dx = sampled[:, 1]
    return torch.stack([dy, dx], dim=1)  # (N, 2)

def target_to_source_points(pts_xy_target, flow_dy_dx):
    """
    pts in target/fixed coords -> corresponding source/moving coords
    Because your flow is backward (target->source): p_src = p_tgt + flow(p_tgt)
    """
    disp_dy_dx = _sample_flow_at_points(flow_dy_dx, pts_xy_target)  # (N, 2) (dy, dx)
    dx = disp_dy_dx[:, 1]
    dy = disp_dy_dx[:, 0]
    return torch.stack([pts_xy_target[:,0] + dx, pts_xy_target[:,1] + dy], dim=1)

def source_to_target_points(pts_xy_source, flow_dy_dx, iters=10):
    """
    pts in source/moving coords -> target/fixed coords.
    We solve q = p_src - flow(q) for q (fixed-point), since flow maps target->source.
    """
    q = pts_xy_source.clone()  # initial guess: target ~= source
    for _ in range(iters):
        disp = _sample_flow_at_points(flow_dy_dx, q)      # (dy, dx) at q
        dx = disp[:, 1]
        dy = disp[:, 0]
        q = torch.stack([pts_xy_source[:,0] - dx, pts_xy_source[:,1] - dy], dim=1)
    return q

def transform_points_xy(points_xy, H):
    """
    points_xy: (N, 2) array of (x, y) in source image coords
    H: (3, 3) homography mapping source -> target
    returns: (N, 2) transformed points in target coords
    """
    pts = np.hstack([points_xy, np.ones((points_xy.shape[0], 1))])  # (N,3)
    pts_t = (H @ pts.T).T                                          # (N,3)
    pts_t = pts_t[:, :2] / pts_t[:, 2:3]
    return pts_t

def get_nonzero_bbox_rgb(image):
    # Detect any non-zero pixel across RGB channels
    nonzero_mask = np.any(image != 0, axis=2)
    rows = np.any(nonzero_mask, axis=1)
    cols = np.any(nonzero_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return xmin, xmax, ymin, ymax


def post_processing_mask_relative(mask: np.ndarray, ratio=0.2) -> np.ndarray:
    mask_filled = ndimage.binary_fill_holes(mask).astype(bool)
    label_img = measure.label(mask_filled)
    props = measure.regionprops(label_img)
    areas = np.array([r.area for r in props])
    if len(areas) == 0:
        return np.zeros_like(mask)
    max_area = areas.max()
    size_thresh = ratio * max_area
    cleaned = morphology.remove_small_objects(label_img, min_size=size_thresh)
    return (cleaned > 0).astype(np.uint8)


def preprocess_images(moving_path, fixed_path, spatial_coord_path, save_dir, global_model_path, local_model_path):
    print(f"Processing ({moving_path}) -> ({fixed_path}).")

    os.makedirs(os.path.join(save_dir, "preprocess_out"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_tissue_seg"), exist_ok=True)
    print("[1] Loading SuperGlue model and image processor...")
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
    model = SuperGlueForKeypointMatching.from_pretrained("magic-leap-community/superglue_outdoor")

    print("[2] Reading input TIFF images...")
    image1 = tiff.imread(moving_path)
    image2 = tiff.imread(fixed_path)

    # Comment out for other images
    image2 = image2[:,:,:3]

    image1_low = cv2.resize(np.array(image1), None, fx=0.1, fy=0.1)
    image2_low = cv2.resize(np.array(image2), None, fx=0.1, fy=0.1)

    image1_mask = tissue_mask_by_chroma(image1_low, blur=7, min_size=8000, close=15, otsu_bias=-30)
    image1_suppressed = suppress_background(image1_low, image1_mask, bg_mode="fill", dim=0.2)  # or bg_mode="fill"
    
    image2_mask = tissue_mask_by_chroma(image2_low, blur=7, min_size=8000, close=15, otsu_bias=-30)
    image2_suppressed = suppress_background(image2_low, image2_mask, bg_mode="fill", dim=0.2)  # or bg_mode="fill"

    print("[3] Running SuperGlue keypoint matching...")
    inputs = processor([[image1_suppressed, image2_suppressed]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    sizes = [[(im.shape[0], im.shape[1]) for im in pair] for pair in [[image1_low, image2_low]]]
    results = processor.post_process_keypoint_matching(outputs, sizes, threshold=0.2)

    kp0 = results[0]['keypoints0']
    kp1 = results[0]['keypoints1']
    print(f"[4] Found {len(kp0)} matched keypoints.")

    print("[5] Estimating homography and warping image1...")
    homography, _ = cv2.findHomography(np.array(kp0), np.array(kp1), cv2.RANSAC)
    # Scale factor
    scale = 0.1
    
    # Create scaling matrix and its inverse
    S = np.diag([1/scale, 1/scale, 1])
    S_inv = np.diag([scale, scale, 1])
    
    # Rescale homography from low-res to high-res
    homography_fullres = S @ homography @ S_inv
    tform = ProjectiveTransform(homography_fullres)




    # Spatial coordinate/landmark load
    spatial_positions = pd.read_csv(spatial_coord_path, header=None)

    spot_ids = spatial_positions[0].values

    spots_xy_fullres = spatial_positions[[5, 4]].to_numpy()

    spots_xy_rigid = transform_points_xy(spots_xy_fullres, homography_fullres)



    
    ##############
    image1_norm = image1.astype(np.float32) / 255.0 if image1.max() > 1.0 else image1.astype(np.float32)
    moving = warp(image1_norm, tform.inverse, output_shape=image2.shape[:2], order=1)
    moving = global_deform.replace_black_with_mode_background(moving)


    fixed = image2.astype(np.float32) / 255.0 if image2.max() > 1.0 else image2.astype(np.float32)

    
    global_model, local_model = load_models(global_model_path, local_model_path)


    ####################
    # Global regisration
    ####################

    factor = 1 / math.ceil(max(fixed.shape) / 1024)
    temp_fixed = cv2.resize(fixed, None, fx=factor, fy=factor).mean(axis=2)
    temp_moving = cv2.resize(moving, None, fx=factor, fy=factor).mean(axis=2)
    
    temp_moving = temp_moving / 255.0 if temp_moving.max() > 1 else temp_moving
    temp_fixed = temp_fixed / 255.0 if temp_fixed.max() > 1 else temp_fixed
    # temp_moving = replace_black_with_mode_background(temp_moving)
    fixed_resized = global_deform.pad_to_1024(temp_fixed)
    moving_resized = global_deform.pad_to_1024(temp_moving)
    
    fixed_tensor = torch.from_numpy(fixed_resized).float().unsqueeze(0)
    moving_tensor = torch.from_numpy(moving_resized).float().unsqueeze(0)
    
    padding_mask = global_deform.create_padding_mask(temp_fixed.shape[0], temp_fixed.shape[1])

    output = global_model(torch.unsqueeze(fixed_tensor, 0), torch.unsqueeze(moving_tensor, 0))
    cropped = global_deform.center_crop(output, temp_fixed.shape[0], temp_fixed.shape[1])

    highres_flow = global_deform.upscale_deformation_field(cropped, new_size=(moving.shape[0], moving.shape[1]))

    print(highres_flow.shape)

    spots_xy_rigid = torch.as_tensor(spots_xy_rigid, dtype=torch.float32, device=highres_flow.device)
    spots_xy_global = source_to_target_points(spots_xy_rigid, highres_flow.squeeze(0), iters=10)  # Nx2

    image_moving_gray_torch = torch.from_numpy(moving.mean(axis=2)).float().unsqueeze(0)
    # print(image_moving_gray_torch.shape)
    warped_high_res = global_deform.apply_deformation_torch(torch.unsqueeze(image_moving_gray_torch, 0), highres_flow)
    np_warped_high_res = warped_high_res.detach().squeeze(0).squeeze(0).numpy()


    ####################
    # Divide patch
    ####################
    patches_fixed = divide_patch.extract_context_patches_with_padding(fixed.mean(axis=2), center_size=512)

    # save_patches_to_folder(patches, save_dir=os.path.join(args.save_dir, "patches_512_registered3_with_global",'fixed'))

    patches_moving = divide_patch.extract_context_patches_with_padding(np_warped_high_res, center_size=512)

    # save_patches_to_folder(patches_moving, save_dir=os.path.join(args.save_dir, "patches_512_registered3_with_global",'moving'))




    ####################
    # Local regisration
    ####################
    shape_orig = fixed.mean(axis=2).shape
    fixed_list = []
    moving_out_list = []
    deformation_field = []
    for i in range(len(patches_fixed)):
        
        moving_patch = patches_moving[i]
        fixed_patch = patches_fixed[i]


        # Normalize
        moving_patch_bw = moving_patch / 255.0 if moving_patch.max() > 1 else moving_patch
        fixed_patch_bw = fixed_patch / 255.0 if fixed_patch.max() > 1 else fixed_patch

        

        # Convert to tensors
        fixed_tensor = torch.from_numpy(fixed_patch_bw).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        moving_tensor = torch.from_numpy(moving_patch_bw).float().unsqueeze(0).unsqueeze(0)

        # fixed_center = center_crop_512x512(fixed_tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
        # fixed_list.append(fixed_center)
        moving_center = local_deform.center_crop_512x512(moving_tensor)

        output = local_model(fixed_tensor, moving_tensor)

        deformation_field.append(output.squeeze(0).squeeze(0).detach().cpu().numpy())

        warped_img = local_deform.apply_deformation_torch(moving_center, output)

        moving_out_list.append(warped_img.squeeze(0).squeeze(0).detach().cpu().numpy())

    print(f'Shapes: {moving_out_list[0].shape}, {deformation_field[0].shape}')

    final_out = local_deform.stitch_patches(moving_out_list, shape_orig, patch_size=512)
    final_field = local_deform.stitch_patches_chw(deformation_field, shape_orig, patch_size=512)
    # filename= f"{path}{f}/final_registered_with_global.tif"
    # tiff.imwrite(filename, final_out)
    tiff.imwrite(os.path.join(save_dir, "preprocess_out/transformed_moving.tif"), final_out)

    # filename2= f"{path}{f}/deformation_field.tif"
    # tiff.imwrite(filename2, final_field)
    tiff.imwrite(os.path.join(save_dir, "preprocess_out/deformation_field.tif"), final_field)
    
    final_field = torch.from_numpy(final_field)
    spots_xy_global = torch.as_tensor(spots_xy_global, dtype=torch.float32, device=final_field.device)
    spots_xy_local = source_to_target_points(spots_xy_global, final_field, iters=10)  # Nx2
    # df = pd.DataFrame(spots_xy_transformed, columns=["x", "y"])
    # df.to_csv("transformed_spots.csv", index=False)

    temp_spots_xy_local = spots_xy_local.detach().numpy()


    # Transformed spatial coordinate/landmark save
    df = pd.DataFrame({
        "barcode": spot_ids,
        "in_tissue": spatial_positions[1].values,
        "row": spatial_positions[2].values,
        "col": spatial_positions[3].values,
        "y_transformed": temp_spots_xy_local[:, 1],
        "x_transformed": temp_spots_xy_local[:, 0]
    })
    # os.path.join(save_dir, "preprocess_out/transformed_spots.csv")
    # df.to_csv(os.path.join(save_dir, "preprocess_out/transformed_spots2.csv"), index=False)

    print(f"{save_dir} Finished.")

    

    




    
    
    # tiff.imwrite(os.path.join(save_dir, "preprocess_out/cropped_moving2.tif"), transformed_img)
    # tiff.imwrite(os.path.join(save_dir, "preprocess_out/cropped_fixed2.tif"), final_cropped_fixed)

    # print(f"[Done] Preprocessing complete. Results saved to: {os.path.join(save_dir, 'preprocess_out')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess paired WSI images using keypoint matching and segmentation.")
    parser.add_argument("--moving_path", type=str, required=True, help="Path to the first image (e.g., H&E).")
    parser.add_argument("--fixed_path", type=str, required=True, help="Path to the second image (e.g., IHC).")
    parser.add_argument("--global_model_path", type=str, required=True, help="Path to the global deformation model.")
    parser.add_argument("--local_model_path", type=str, required=True, help="Path to the local deformation model.")
    parser.add_argument("--spatial_coord_path", type=str, required=True, help="Path to the spatial coordinate file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the processed outputs.")

    args = parser.parse_args()
    preprocess_images(args.moving_path, args.fixed_path, args.spatial_coord_path, args.save_dir, args.global_model_path, args.local_model_path)
