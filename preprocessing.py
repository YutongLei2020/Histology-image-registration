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


def preprocess_images(data1_path, data2_path, save_dir):
    os.makedirs(os.path.join(save_dir, "preprocess_out"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_tissue_seg"), exist_ok=True)
    print("[1] Loading SuperGlue model and image processor...")
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
    model = SuperGlueForKeypointMatching.from_pretrained("magic-leap-community/superglue_outdoor")

    print("[2] Reading input TIFF images...")
    image1 = tiff.imread(data1_path)
    image2 = tiff.imread(data2_path)

    image1_low = cv2.resize(np.array(image1), None, fx=0.1, fy=0.1)
    image2_low = cv2.resize(np.array(image2), None, fx=0.1, fy=0.1)

    print("[3] Running SuperGlue keypoint matching...")
    inputs = processor([[image1_low, image2_low]], return_tensors="pt")
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

    image1_norm = image1.astype(np.float32) / 255.0 if image1.max() > 1.0 else image1.astype(np.float32)
    transformed_img = warp(image1_norm, tform.inverse, output_shape=image2.shape[:2], order=1)

    print("[6] Creating Otsu masks from downsampled preview...")
    temp = cv2.resize(transformed_img, None, fx=0.1, fy=0.1)
    temp2 = image2_low
    # mask_trans = create_otsu_mask(temp)
    # mask_fixed = create_otsu_mask(temp2)

    # joint_mask = mask_trans & mask_fixed
    # ys, xs = np.where(joint_mask)
    # if len(xs) == 0 or len(ys) == 0:
    #     raise ValueError("No overlap found")
    print("[7] Computing bounding box of overlapping area...")
    # xmin, xmax = xs.min(), xs.max()
    # ymin, ymax = ys.min(), ys.max()
    xmin, xmax, ymin, ymax = get_nonzero_bbox_rgb(temp)
    factor = 10
    xmin_o, xmax_o = xmin * factor, xmax * factor
    ymin_o, ymax_o = ymin * factor, ymax * factor

    cropped_trans_orig = transformed_img[ymin_o:ymax_o + 1, xmin_o:xmax_o + 1]
    cropped_fixed_orig = image2[ymin_o:ymax_o + 1, xmin_o:xmax_o + 1]

    print("[8] Saving preview images for tissue segmentation...")
    cv2.imwrite(os.path.join(save_dir, "before_tissue_seg/cropped_fixed.png"), temp2[ymin:ymax+1, xmin:xmax+1])
    cv2.imwrite(os.path.join(save_dir, "before_tissue_seg/cropped_moving.png"), (temp[ymin:ymax+1, xmin:xmax+1] * 255).astype(np.uint8))

    print("[9] Running tissue segmentation using TIAToolbox...")
    segmentor = SemanticSegmentor(pretrained_model="unet_tissue_mask_tsef", num_loader_workers=4, batch_size=4)
    output = segmentor.predict([save_dir+"before_tissue_seg/cropped_fixed.png", save_dir+"before_tissue_seg/cropped_moving.png"], mode="tile",
                                save_dir=os.path.join(save_dir, "tissue_seg_out"))

    print("[10] Loading and processing predicted tissue masks...")
    fixed_mask = np.argmax(np.load(output[0][1] + ".raw.0.npy"), axis=-1) == 2
    moving_mask = np.argmax(np.load(output[1][1] + ".raw.0.npy"), axis=-1) == 2
    fixed_mask = post_processing_mask_relative(fixed_mask)
    moving_mask = post_processing_mask_relative(moving_mask)

    print("[11] Resizing masks and determining final crop area...")
    scale = 10
    fixed_mask = cv2.resize(fixed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    moving_mask = cv2.resize(moving_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    joint_mask = fixed_mask | moving_mask
    ys, xs = np.where(joint_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No overlapping content in masks")

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    print("[12] Cropping and saving final output TIFF images...")
    final_cropped_moving = cropped_trans_orig[ymin:ymax+1, xmin:xmax+1]
    final_cropped_fixed = cropped_fixed_orig[ymin:ymax+1, xmin:xmax+1]

    tiff.imwrite(os.path.join(save_dir, "preprocess_out/cropped_moving.tif"), final_cropped_moving)
    tiff.imwrite(os.path.join(save_dir, "preprocess_out/cropped_fixed.tif"), final_cropped_fixed)

    print(f"[Done] Preprocessing complete. Results saved to: {os.path.join(save_dir, 'preprocess_out')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess paired WSI images using keypoint matching and segmentation.")
    parser.add_argument("--data1_path", type=str, required=True, help="Path to the first image (e.g., H&E).")
    parser.add_argument("--data2_path", type=str, required=True, help="Path to the second image (e.g., IHC).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the processed outputs.")

    args = parser.parse_args()
    preprocess_images(args.data1_path, args.data2_path, args.save_dir)
