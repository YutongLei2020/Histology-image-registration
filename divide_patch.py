import argparse
import os
import numpy as np
import torch
import tifffile as tiff
import cv2
import imageio.v3 as imageio
from skimage.filters import threshold_otsu
from skimage.transform import warp, ProjectiveTransform
from skimage import measure, morphology
from scipy import ndimage
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

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

def extract_context_patches_with_padding(img: np.ndarray, center_size=1024, context=128):
    patch_size = center_size + 2 * context
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    num_patches_y = int(np.ceil(H / center_size))
    num_patches_x = int(np.ceil(W / center_size))
    padded_H = num_patches_y * center_size
    padded_W = num_patches_x * center_size
    pad_bottom = padded_H - H
    pad_right = padded_W - W

    if C == 1:
        img_padded = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='reflect')
    else:
        img_padded = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')

    if C == 1:
        img_padded = np.pad(img_padded, ((context, context), (context, context)), mode='reflect')
    else:
        img_padded = np.pad(img_padded, ((context, context), (context, context), (0, 0)), mode='reflect')

    patches = []
    for i in range(0, padded_H, center_size):
        for j in range(0, padded_W, center_size):
            patch = img_padded[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return np.array(patches)

def save_patches_to_folder(patches: np.ndarray, save_dir: str, prefix='patch'):
    os.makedirs(save_dir, exist_ok=True)
    for idx, patch in enumerate(patches):
        if patch.dtype != np.uint8:
            patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
        filename = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")
        imageio.imwrite(filename, patch)
    print(f"Saved {len(patches)} patches to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Context-aware patch extraction with tissue segmentation")
    parser.add_argument('--input_fixed', required=True, help='Path to fixed image TIFF')
    parser.add_argument('--input_moving', required=True, help='Path to moving image TIFF')
    parser.add_argument('--save_dir', required=True, help='Output directory for cropped and patch data')
    args = parser.parse_args()
    print('sdfsdfsdfsdfsdf:', os.path.join(args.save_dir, "tissue_seg"))

    os.makedirs(os.path.join(args.save_dir, "tissue_seg"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "tissue_seg", "before_tissue_seg"), exist_ok=True)

    print("[1] Loading input images...")
    cropped_fixed_orig = tiff.imread(args.input_fixed)
    cropped_trans_orig = tiff.imread(args.input_moving)

    transformed_img = cropped_trans_orig.astype(np.float32) / 255.0
    # transformed_img = cv2.resize(transformed_img, None, fx=0.5, fy=0.5) # temp
    image2 = cropped_fixed_orig.astype(np.float32) / 255.0
    # image2 = cv2.resize(image2, None, fx=0.5, fy=0.5)
    print('Shapes: ',transformed_img.shape, image2.shape)

    temp = cv2.resize(transformed_img, None, fx=0.1, fy=0.1)
    temp2 = cv2.resize(image2, None, fx=0.1, fy=0.1)

    fixed_png = os.path.join(args.save_dir,'tissue_seg', "before_tissue_seg", "cropped_fixed.png")
    moving_png = os.path.join(args.save_dir,'tissue_seg', "before_tissue_seg", "cropped_moving.png")
    cv2.imwrite(fixed_png, (temp2 * 255).astype(np.uint8))
    cv2.imwrite(moving_png, (temp * 255).astype(np.uint8))

    print("[2] Running tissue segmentation using TIAToolbox...")
    segmentor = SemanticSegmentor(pretrained_model="unet_tissue_mask_tsef", num_loader_workers=4, batch_size=4)
    output = segmentor.predict([fixed_png, moving_png], mode="tile", save_dir=os.path.join(args.save_dir,'tissue_seg', "tissue_seg_out"))
    # output = [['',args.save_dir+'/tissue_seg'+"/tissue_seg_out/0"],
    #           ['',args.save_dir+'/tissue_seg'+"/tissue_seg_out/1"]]
    
    print("[3] Processing tissue masks...")
    fixed_mask = np.argmax(np.load(output[0][1] + ".raw.0.npy"), axis=-1) == 2
    moving_mask = np.argmax(np.load(output[1][1] + ".raw.0.npy"), axis=-1) == 2

    fixed_mask = post_processing_mask_relative(fixed_mask)
    moving_mask = post_processing_mask_relative(moving_mask)

    scale = 10
    fixed_mask = cv2.resize(fixed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    moving_mask = cv2.resize(moving_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    joint_mask = fixed_mask | moving_mask
    ys, xs = np.where(joint_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No overlapping content in tissue masks")

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    print("[4] Cropping final images...")
    final_cropped_moving = transformed_img[ymin:ymax+1, xmin:xmax+1]
    final_cropped_fixed = image2[ymin:ymax+1, xmin:xmax+1]

    print("[5] Extracting context-aware patches from fixed image...")
    patches = extract_context_patches_with_padding(final_cropped_fixed, center_size=512)

    print("[6] Saving patches...")
    save_patches_to_folder(patches, save_dir=os.path.join(args.save_dir, "patches_512",'fixed'))

    print("[7] Extracting context-aware patches from moving image...")
    patches_moving = extract_context_patches_with_padding(final_cropped_moving, center_size=512)

    print("[8] Saving patches...")
    save_patches_to_folder(patches_moving, save_dir=os.path.join(args.save_dir, "patches_512",'moving'))

if __name__ == '__main__':
    main()
