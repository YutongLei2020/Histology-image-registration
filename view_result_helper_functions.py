import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
import SimpleITK as sitk
import os
import torch
from torchvision import transforms
import numpy as np
import tifffile as tiff

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
import math
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
import pickle
from scipy import stats
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image
import json
import statistics
import math
from math import pi
from scipy.spatial import cKDTree
import torch
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2
from skimage import color, filters, exposure, morphology, measure


def plot_two_visium_spot_sets(
    pos1, pos2, 
    radius1, radius2,
    color1="red", color2="blue",
    alpha=0.4,
    linewidth=0.6,
    tissue_only=True,
    img_shape=None  # optional if you don’t want background image
):
    """
    pos1, pos2: DataFrames with 'x','y','in_tissue' (like tissue_positions_list.csv)
    radius1, radius2: radii for spots in pixel units
    color1, color2: circle colors
    img_shape: (H, W) of background, if no image is shown
    """

    if tissue_only:
        pos1 = pos1[pos1["in_tissue"] == 1].copy()
        pos2 = pos2[pos2["in_tissue"] == 1].copy()

    # Background canvas
    if img_shape is None:
        H = max(pos1["y"].max(), pos2["y"].max()) + 100
        W = max(pos1["x"].max(), pos2["x"].max()) + 100
    else:
        H, W = img_shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])  # image-style coords (0,0 top left)

    # Draw circles for dataset 1
    for _, r in pos1.iterrows():
        circ = Circle(
            (r["x"], r["y"]),
            radius=radius1,
            ec=color1,
            fc=color1,
            lw=linewidth,
            alpha=alpha
        )
        ax.add_patch(circ)

    # Draw circles for dataset 2
    for _, r in pos2.iterrows():
        circ = Circle(
            (r["x"], r["y"]),
            radius=radius2,
            ec=color2,
            fc=color2,
            lw=linewidth,
            alpha=alpha
        )
        ax.add_patch(circ)

    ax.set_title("Two Visium spot sets overlay")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
def plot_images_with_coords(images, coords, canvas_shape, patch_size=512):
    """
    images: list of np arrays (H, W) or (H, W, C), all same size
    coords: list of (x, y) top-left coordinates for each patch
    canvas_shape: (H, W) of the full stitched canvas
    patch_size: size of each square patch
    """
    # Initialize empty canvas
    if images[0].ndim == 2:  # grayscale
        canvas = np.zeros(canvas_shape, dtype=images[0].dtype)
    else:  # color
        C = images[0].shape[2]
        canvas = np.zeros((*canvas_shape, C), dtype=images[0].dtype)

    # Place each image patch on the canvas
    for img, (x, y) in zip(images, coords):
        canvas[y:y+patch_size, x:x+patch_size] = img

    # Plot result
    plt.figure(figsize=(10, 10))
    if canvas.ndim == 2:
        plt.imshow(canvas, cmap="gray")
    else:
        plt.imshow(canvas)
    plt.axis("off")
    plt.show()

    return canvas

from scipy.ndimage import binary_fill_holes

def mask_he_structure1(
    img_bgr,
    channel="E",              # "H" (hematoxylin) or "E" (eosin) or "HE" (blend)
    clip_limit=0.01,          # contrast stretch (0 disables)
    blur_sigma=1.0,           # light smoothing before threshold
    thresh_mode="otsu",       # "otsu" or "adaptive"
    adaptive_block=151,       # odd window size for adaptive thresh
    adaptive_C=0,             # subtracted constant for adaptive
    min_area=2000,            # remove tiny debris (pixels)
    close_size=9,             # closing kernel (odd)
    open_size=5,              # opening kernel (odd)
    fill_holes=True,          # fill internal holes
    keep_largest=False,       # keep only largest component
    roi_polygon=None          # optional Nx2 (x,y) polygon to restrict search
):
    """
    img_bgr: uint8 HxWx3 (OpenCV read). If you have RGB, swap first (cv2.cvtColor).
    Returns: binary mask uint8 {0,1}
    """
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "img_bgr must be HxWx3 BGR"
    H, W = img_bgr.shape[:2]

    # Optional ROI restriction (everything outside set to background before processing)
    roi_mask = None
    if roi_polygon is not None:
        roi_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(roi_mask, [np.asarray(roi_polygon, np.int32)], 1)

    # --- 1) Color deconvolution (RGB → HED), then pick channel(s)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # skimage returns float; H,E,D roughly correspond to nuclei, cytoplasm, DAB.
    hed = color.rgb2hed(rgb)

    Hc = hed[..., 0]      # hematoxylin (nuclei) — more NEGATIVE = stronger stain
    Ec = hed[..., 1]      # eosin      (cytoplasm/extracellular) — more NEGATIVE = stronger
    # Normalize each to [0,1] with sign flipped so "more stained = higher"
    Hn = (-(Hc) - (-(Hc)).min()) / ((-(Hc)).ptp() + 1e-8)
    En = (-(Ec) - (-(Ec)).min()) / ((-(Ec)).ptp() + 1e-8)

    if channel == "H":
        img_scalar = Hn
    elif channel == "E":
        img_scalar = En
    elif channel == "HE":
        img_scalar = 0.6 * En + 0.4 * Hn
    else:
        raise ValueError("channel must be 'H', 'E', or 'HE'")

    # Optional: restrict to ROI early
    if roi_mask is not None:
        img_scalar = img_scalar * roi_mask

    # Light smoothing + contrast stretch
    if blur_sigma and blur_sigma > 0:
        img_scalar = cv2.GaussianBlur(img_scalar, (0, 0), blur_sigma)
    if clip_limit and clip_limit > 0:
        lo, hi = np.quantile(img_scalar[img_scalar > 0], [clip_limit, 1 - clip_limit]) if np.any(img_scalar > 0) else (0,1)
        img_scalar = np.clip((img_scalar - lo) / (hi - lo + 1e-8), 0, 1)

    # --- 2) Threshold
    if thresh_mode == "otsu":
        t = filters.threshold_otsu(img_scalar[img_scalar > 0]) if np.any(img_scalar > 0) else 0.5
        binmask = (img_scalar >= t).astype(np.uint8)
    elif thresh_mode == "adaptive":
        # adaptiveThreshold needs uint8 0..255
        u8 = np.uint8(np.clip(img_scalar * 255, 0, 255))
        binmask = cv2.adaptiveThreshold(
            u8, 255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=adaptive_block, C=adaptive_C
        ) // 255
    else:
        raise ValueError("thresh_mode must be 'otsu' or 'adaptive'")

    # print(1)
    # return binmask 
    # print(2)
        
    # Optional ROI hard-limit after threshold
    if roi_mask is not None:
        binmask = binmask * roi_mask

    # --- 3) Clean-up (morphology, area filter, holes)
    # remove tiny components
    # if min_area and min_area > 0:
    #     lbl = measure.label(binmask, connectivity=2)
    #     counts = np.bincount(lbl.ravel())
    #     keep = np.where(counts >= min_area)[0]
    #     keep_mask = np.isin(lbl, keep).astype(np.uint8)
    #     binmask = keep_mask
    # return binmask 
    # closing (fill gaps) then opening (remove spurs)
    if close_size and close_size > 1:
        k = np.ones((close_size | 1, close_size | 1), np.uint8)
        binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, k)
    if open_size and open_size > 1:
        k = np.ones((open_size | 1, open_size | 1), np.uint8)
        binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, k)
    # return binmask 
    binmask = 1 - binmask
    if fill_holes:
        # flood fill from border to find exterior, then invert to fill inner holes
        ff = (binmask == 0).astype(np.uint8) * 255
        h, w = binmask.shape
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask_ff, (0, 0), 128)  # mark exterior as 128
        holes = (ff != 128).astype(np.uint8)     # interior holes were 0
        binmask = np.maximum(binmask, holes)

        # # binmask: uint8 {0,1}, foreground = 1
        # inv = (1 - binmask).astype(np.uint8) * 255   # background=255, foreground=0
        
        # h, w = inv.shape
        # mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        
        # # mark EXTERIOR background connected to the border as 128
        # cv2.floodFill(inv, mask_ff, (0, 0), 128)
        
        # # holes are background pixels NOT connected to the border (still 255)
        # holes = (inv == 255).astype(np.uint8)
        
        # # fill only those holes
        # binmask = np.maximum(binmask, holes)  # foreground stays 1; holes become 1



        # binmask = binary_fill_holes(binmask.astype(bool)).astype(np.uint8)

    # return binmask 
    # binmask = 1 - binmask
    if keep_largest:
        lbl = measure.label(binmask, connectivity=2)
        if lbl.max() > 0:
            areas = np.bincount(lbl.ravel())[1:]
            idx = np.argmax(areas) + 1
            binmask = (lbl == idx).astype(np.uint8)

    return binmask  # uint8 {0,1}


def plot_spot_coords(coords, colors=None, size=30, title="Spot Coordinates"):
    """
    coords: (N,2) array-like of (x,y) coordinates
    colors: optional (N,) array-like for coloring each dot
    size: marker size
    """
    coords = np.asarray(coords)
    x, y = coords[:,0], coords[:,1]

    plt.figure(figsize=(8,8))
    if colors is None:
        plt.scatter(x, y, s=size, c="blue", alpha=0.7)
    else:
        plt.scatter(x, y, s=size, c=colors, alpha=0.7, cmap="viridis")

    plt.gca().invert_yaxis()   # keep image-style coords (origin at top-left)
    plt.axis("equal")
    plt.title(title)
    plt.show()

def circle_intersection_area(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return pi * min(r1, r2)**2
    r1_2, r2_2 = r1*r1, r2*r2
    alpha = np.arccos((d*d + r1_2 - r2_2) / (2*d*r1))
    beta  = np.arccos((d*d + r2_2 - r1_2) / (2*d*r2))
    return r1_2*alpha + r2_2*beta - 0.5*np.sqrt(
        max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
    )

def compute_spot_overlaps_variable_radii(
    fixed_xy,                    # (N,2) in target pixel coords (e.g., fixed image)
    moving_xy,                   # (M,2) transformed into the same pixel coords
    r_fixed,                     # scalar or (N,) radii in target pixels
    r_moving,                    # scalar or (M,) radii in target pixels
    fixed_ids=None,              # optional (N,)
    moving_ids=None,             # optional (M,)
    csv_out=None
):
    fixed_xy  = np.asarray(fixed_xy, dtype=float)
    moving_xy = np.asarray(moving_xy, dtype=float)

    # Broadcast radii
    r_fixed  = np.asarray(r_fixed,  dtype=float)
    r_moving = np.asarray(r_moving, dtype=float)
    if r_fixed.ndim == 0:
        r_fixed = np.full(fixed_xy.shape[0], r_fixed, dtype=float)
    if r_moving.ndim == 0:
        r_moving = np.full(moving_xy.shape[0], r_moving, dtype=float)

    if fixed_ids is None:
        fixed_ids = np.arange(fixed_xy.shape[0])
    if moving_ids is None:
        moving_ids = np.arange(moving_xy.shape[0])

    tree = cKDTree(fixed_xy)
    records = []

    # Loop moving spots and query candidates with per-spot search radius
    for j, (xj, yj) in enumerate(moving_xy):
        # Max possible distance for overlap for this moving spot
        search_r = (r_fixed.max() + r_moving[j])  # conservative; you can speed up by binning r_fixed
        idxs = tree.query_ball_point([xj, yj], search_r)
        area_moving = pi * (r_moving[j]**2)

        for i in idxs:
            xi, yi = fixed_xy[i]
            d = np.hypot(xi - xj, yi - yj)
            inter = circle_intersection_area(d, r_fixed[i], r_moving[j])
            if inter <= 0:
                continue
            area_fixed = pi * (r_fixed[i]**2)
            overlap_fixed_pct  = inter / area_fixed
            overlap_moving_pct = inter / area_moving
            iou = inter / (area_fixed + area_moving - inter)

            records.append({
                "fixed_id":  fixed_ids[i],
                "moving_id": moving_ids[j],
                "fixed_x":   xi, "fixed_y":  yi,
                "moving_x":  xj, "moving_y": yj,
                "r_fixed":   r_fixed[i],
                "r_moving":  r_moving[j],
                "distance_px": d,
                "intersection_area": inter,
                "overlap_fixed_pct":  overlap_fixed_pct,
                "overlap_moving_pct": overlap_moving_pct,
                "iou": iou
            })

    df = pd.DataFrame.from_records(records)
    if csv_out:
        df.to_csv(csv_out, index=False)
    return df

def ncc_loss(warped, target, eps=1e-5):
    # If input is numpy array, convert to torch tensor
    if isinstance(warped, np.ndarray):
        warped = torch.from_numpy(warped).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    # Ensure tensors have shape (B, C, H, W)
    if warped.ndim == 2:  # (H, W)
        warped = warped.unsqueeze(0).unsqueeze(0)
    elif warped.ndim == 3:  # (C, H, W)
        # print(1)
        # warped = np.transpose(warped, (2, 0, 1))
        warped = warped.permute(2,0,1)
        warped = warped.unsqueeze(0)

    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        # target = np.transpose(target, (2, 0, 1))
        target = target.permute(2,0,1)
        target = target.unsqueeze(0)

    mean_warped = warped.mean(dim=[2, 3], keepdim=True)
    mean_target = target.mean(dim=[2, 3], keepdim=True)

    warped_centered = warped - mean_warped
    target_centered = target - mean_target

    numerator = (warped_centered * target_centered).mean(dim=[2, 3])
    denominator = (warped_centered.pow(2).mean(dim=[2, 3]) *
                   target_centered.pow(2).mean(dim=[2, 3])).sqrt() + eps

    ncc = numerator / denominator
    return -ncc.mean()

def replace_black_with_mode_background(image):
    """
    Replace black (zero) pixels with the mode of nonzero pixels.
    Works best when background dominates the image.
    """
    nonzero_pixels = image[image > 0]
    mode_val = stats.mode(nonzero_pixels, axis=None, keepdims=False).mode
    image_filled = image.copy()
    image_filled[image == 0] = mode_val
    return image_filled
def show_magnitude(flow, title="Flow magnitude"):
    if flow.ndim == 4: flow = flow[0]
    if isinstance(flow, torch.Tensor): flow = flow.detach().cpu().numpy()
    dy, dx = flow
    mag = np.sqrt(dx**2 + dy**2)

    plt.figure(figsize=(6,5))
    plt.imshow(mag, cmap='magma')
    plt.colorbar(label='pixels')
    plt.title(title); plt.axis('off'); plt.tight_layout(); plt.show()

def show_warped_grid(flow, step=32, color='c'):
    if flow.ndim == 4: flow = flow[0]
    if isinstance(flow, torch.Tensor): flow = flow.detach().cpu().numpy()
    dy, dx = flow
    H, W = dy.shape

    yy, xx = np.mgrid[0:H:step, 0:W:step]
    xx_w = xx + dx[::step, ::step]
    yy_w = yy + dy[::step, ::step]

    plt.figure(figsize=(6,6))
    # draw horizontal lines
    for r in range(yy_w.shape[0]):
        plt.plot(xx_w[r, :], yy_w[r, :], color=color, linewidth=1)
    # draw vertical lines
    for c in range(xx_w.shape[1]):
        plt.plot(xx_w[:, c], yy_w[:, c], color=color, linewidth=1)

    plt.gca().invert_yaxis()
    plt.axis('equal'); plt.title("Warped grid"); plt.tight_layout(); plt.show()
def to_grayscale(img):
    # img: uint8 HxWx3 or float; returns float32 [0,1]
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / (255.0 if gray.dtype == np.uint8 else np.max(gray)+1e-8)
    return gray

def tissue_mask_from_gray(gray, blur=3, thresh=0.3):
    """ crude mask: anything darker than high intensity (i.e., not bright background) """
    g = cv2.GaussianBlur(gray, (0,0), blur)
    # invert so tissue (darker) ~ high; background (bright) ~ low
    inv = 1.0 - g
    m = (inv > thresh).astype(np.uint8)
    # clean up
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return m

def edge_strength(gray, sigma=1.0):
    """ Sobel edges on lightly smoothed image """
    g = cv2.GaussianBlur(gray, (0,0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = mag / (mag.max() + 1e-8)
    return mag

def build_weight_map(img, w_edge=0.7, w_bright=0.3, rim_boost_px=25, rim_gain=0.5, thresh=0.3):
    """
    Content-aware weight 0..1:
      - stronger near edges (w_edge)
      - stronger in bright/low-density regions (w_bright)
      - optional boost near tissue rim (rim_gain within rim_boost_px)
    """
    H, W = img.shape[:2]
    gray = to_grayscale(img)
    mask = tissue_mask_from_gray(gray, thresh)  # 1 = tissue, 0 = background

    E = edge_strength(gray, sigma=1.0)         # 0..1 (edges high)
    bright = gray                               # 0..1 (bright high)
    bright = (bright - bright.min()) / (bright.max() - bright.min() + 1e-8)

    # distance to background (inside tissue)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
    rim = np.clip(1.0 - dist_in / float(rim_boost_px), 0, 1)  # 1 at edge → 0 inside

    # combine
    w = w_edge * E + w_bright * bright + rim_gain * rim
    w *= mask.astype(np.float32)
    # smooth to avoid speckle; keep in 0..1
    w = cv2.GaussianBlur(w, (0,0), 3.0)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    return w, mask
def sample_bspline_field(H, W, grid=(12,12), max_disp=8.0, smooth=1.5, seed=None):
    rng = np.random.default_rng(seed)
    gy, gx = grid
    ctrl = rng.uniform(-max_disp, max_disp, size=(2, gy, gx)).astype(np.float32)  # [dy, dx]

    yy = np.linspace(0, gy-1, H, dtype=np.float32)
    xx = np.linspace(0, gx-1, W, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')

    dy = map_coordinates(ctrl[0], [grid_y, grid_x], order=3, mode='reflect')
    dx = map_coordinates(ctrl[1], [grid_y, grid_x], order=3, mode='reflect')

    if smooth and smooth > 0:
        dy = gaussian_filter(dy, smooth)
        dx = gaussian_filter(dx, smooth)

    return np.stack([dy, dx], axis=-1)  # (H,W,2)

def make_content_aware_field(img, 
                             grid=(12,12), max_disp=8.0, smooth=1.5,
                             w_edge=0.7, w_bright=0.3, rim_boost_px=25, rim_gain=0.5,
                             overall_gain=1.0, seed=None):
    """
    1) Build weight map from image structure
    2) Sample a smooth base field
    3) Modulate base field by weight, zero outside tissue
    """
    H, W = img.shape[:2]
    weight, mask = build_weight_map(img, w_edge, w_bright, rim_boost_px, rim_gain)  # 0..1
    base = sample_bspline_field(H, W, grid, max_disp, smooth, seed=seed)            # [dy,dx]

    weight3 = weight[..., None]  # broadcast to 2 channels
    disp = overall_gain * base * weight3
    disp *= mask[..., None].astype(np.float32)  # zero-out background
    return disp.astype(np.float32), weight, mask
def warp_image_with_field(img, disp):
    H, W = disp.shape[:2]
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')
    map_x = (xx + disp[..., 1]).astype(np.float32)
    map_y = (yy + disp[..., 0]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101)

def warp_points_with_field(points, disp):
    H, W = disp.shape[:2]
    x = np.clip(points[:, 0], 0, W-1)
    y = np.clip(points[:, 1], 0, H-1)

    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.clip(x0+1, 0, W-1);   y1 = np.clip(y0+1, 0, H-1)

    wa = (x1-x)*(y1-y); wb = (x-x0)*(y1-y); wc = (x1-x)*(y-y0); wd = (x-x0)*(y-y0)

    dy = (wa*disp[y0, x0, 0] + wb*disp[y0, x1, 0] +
          wc*disp[y1, x0, 0] + wd*disp[y1, x1, 0])
    dx = (wa*disp[y0, x0, 1] + wb*disp[y0, x1, 1] +
          wc*disp[y1, x0, 1] + wd*disp[y1, x1, 1])

    return np.column_stack([x+dx, y+dy]).astype(np.float32)

import matplotlib.pyplot as plt

def show_deformation_quiver(disp, step=50, scale=1.0):
    """
    disp: (H,W,2) with [dy,dx]
    step: sample spacing (larger = fewer arrows)
    scale: scaling factor for arrow length
    """
    H, W = disp.shape[:2]
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    u = disp[::step, ::step, 1] * scale  # dx
    v = disp[::step, ::step, 0] * scale  # dy

    plt.figure(figsize=(6,6))
    plt.quiver(xx, yy, u, v, angles='xy', scale_units='xy', scale=1, color='r')
    plt.gca().invert_yaxis()
    plt.title("Deformation field (quiver)")
    plt.show()

import numpy as np, cv2

def rim_directional_drag_field(mask,  # uint8 {0,1} tissue mask
                               direction=(8.0, -4.0),  # (dx, dy) pixels; right=+x, down=+y
                               band_px=80,             # width of rim band
                               inner_taper_px=30,      # how smoothly it fades inside
                               edge_taper_sigma=8.0,   # blur of the weight
                               jitter_frac=0.1,        # 0..0.3 small randomness (0 = none)
                               seed=None):
    """
    Returns disp (H,W,2) with [dy, dx] that drags the tissue rim in a common direction.
    """
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    # distance from background (inside tissue)
    dist_in = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    # rim weight: 1 at edge, ~0 past band_px
    rim = np.clip(1.0 - dist_in / float(band_px), 0, 1)

    # optional inner taper so it dies smoothly toward interior
    if inner_taper_px > 0:
        core = np.clip(dist_in / float(inner_taper_px), 0, 1)
        rim = rim * (1.0 - core) + rim * 0.0  # extra softness near the edge

    # smooth the weight to avoid hard borders
    rim = cv2.GaussianBlur(rim, (0,0), edge_taper_sigma)

    # base constant vector
    dx0, dy0 = direction
    disp_x = rim * dx0
    disp_y = rim * dy0

    # tiny spatially-smooth jitter so it's not perfectly uniform
    if jitter_frac > 0:
        jx = cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32), (0,0), 12)
        jy = cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32), (0,0), 12)
        jx = jx / (np.max(np.abs(jx)) + 1e-6)
        jy = jy / (np.max(np.abs(jy)) + 1e-6)
        disp_x += jitter_frac * dx0 * rim * jx
        disp_y += jitter_frac * dy0 * rim * jy

    # zero outside tissue
    disp_x *= mask.astype(np.float32)
    disp_y *= mask.astype(np.float32)

    return np.stack([disp_y, disp_x], axis=-1).astype(np.float32)

def regional_directional_shove_field(H, W,
                                     roi_mask,             # uint8 {0,1} region to push
                                     direction=(12.0, 0.0),# (dx,dy) pixels
                                     feather_px=60,        # feathering width
                                     jitter_frac=0.1):
    """
    Push a chosen region in a common direction with smooth feathered edges.
    """
    # feather the ROI into a soft weight
    weight = cv2.GaussianBlur(roi_mask.astype(np.float32), (0,0), feather_px)
    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-6)

    dx0, dy0 = direction
    disp_x = weight * dx0
    disp_y = weight * dy0

    if jitter_frac > 0:
        rng = np.random.default_rng(123)
        jx = cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32), (0,0), 10)
        jy = cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32), (0,0), 10)
        jx /= (np.max(np.abs(jx)) + 1e-6)
        jy /= (np.max(np.abs(jy)) + 1e-6)
        disp_x += jitter_frac * dx0 * weight * jx
        disp_y += jitter_frac * dy0 * weight * jy

    return np.stack([disp_y, disp_x], axis=-1).astype(np.float32)
import numpy as np, cv2

def directional_roi_field(
    tissue_mask,          # uint8 {0,1} full tissue mask
    roi_partial_mask,     # uint8 {0,1} "partial" region you drew/selected
    base_direction=(10.0, -6.0),  # (dx, dy) pixels; right=+x, down=+y
    magnitude_px=None,    # if None, |base_direction| is used
    max_angle_deg=12.0,   # max local deviation from base dir (smoothly)
    jitter_sigma_px=120,  # smoothness of direction jitter (bigger = smoother)
    feather_px=80,        # soft fade at ROI border
    fill_holes=True,      # close gaps inside your partial ROI
    close_kernel=15,      # morphological closing kernel size
    seed=None
):
    """
    Returns disp (H,W,2) with [dy,dx] that pushes a *filled+feathered* ROI mostly
    in one direction, with smooth, small directional variation.
    """
    rng = np.random.default_rng(seed)
    H, W = tissue_mask.shape

    # --- 1) Build a *dense* ROI from your partial mask (so all pixels inside move)
    roi = (roi_partial_mask.astype(np.uint8) & tissue_mask.astype(np.uint8))
    if fill_holes:
        # close small gaps
        k = max(3, int(close_kernel) | 1)  # odd
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        # fill holes (flood-fill from border)
        ff = roi.copy()
        cv2.floodFill(ff, np.zeros((H+2, W+2), np.uint8), (0, 0), 255)
        holes = (ff == 0).astype(np.uint8)  # interior holes
        roi = np.maximum(roi, holes) & tissue_mask

    # --- 2) Make a SOFT weight so it fades at the border (no hard stop)
    # feather via Gaussian; normalize to [0,1]
    soft = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), feather_px)
    if soft.max() > 0:
        soft = (soft - soft.min()) / (soft.max() - soft.min() + 1e-6)
    soft *= tissue_mask.astype(np.float32)  # keep inside tissue

    # --- 3) Create a *smooth* direction field around your base direction
    # base unit vector
    dx0, dy0 = base_direction
    mag0 = np.hypot(dx0, dy0) + 1e-6
    u0x, u0y = dx0 / mag0, dy0 / mag0

    # smooth random 2-channel jitter
    jx = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0,0), jitter_sigma_px)
    jy = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0,0), jitter_sigma_px)
    # normalize jitter direction (avoid zero)
    norm = np.sqrt(jx*jx + jy*jy) + 1e-6
    jx /= norm; jy /= norm

    # convert max angle to mixing weight in the plane
    # small-angle approximation: tan(theta) ≈ theta (rad)
    theta = np.deg2rad(max_angle_deg)
    eps = np.tan(theta)  # how much we allow deviation

    # combine: dir = normalize( u0 + eps * jitter )
    dir_x = u0x + eps * jx
    dir_y = u0y + eps * jy
    dnorm = np.sqrt(dir_x*dir_x + dir_y*dir_y) + 1e-6
    dir_x /= dnorm; dir_y /= dnorm

    # --- 4) Choose magnitude inside ROI (constant or inherited)
    if magnitude_px is None:
        magnitude_px = mag0  # default: match base_direction magnitude

    # final displacement, smoothly faded at ROI boundary
    disp_x = soft * magnitude_px * dir_x
    disp_y = soft * magnitude_px * dir_y

    # zero outside tissue
    disp_x *= tissue_mask.astype(np.float32)
    disp_y *= tissue_mask.astype(np.float32)

    return np.stack([disp_y, disp_x], axis=-1).astype(np.float32), soft

import numpy as np, cv2

import numpy as np, cv2

def directional_roi_field_gradient(
    tissue_mask,            # uint8 {0,1} full tissue mask
    roi_partial_mask,       # uint8 {0,1} your drawn/sparse ROI
    base_direction=(12.0, -6.0),   # (dx,dy) pixels; right=+x, down=+y
    magnitude_px=None,      # None -> |base_direction|
    angle_top_deg=12.0,     # rotation at the "start" side (e.g., top or left)
    angle_bottom_deg=2.0,   # rotation at the "end" side (e.g., bottom or right)
    axis='vertical',        # 'vertical' (top->bottom gradient) or 'horizontal'
    feather_px=90,          # soft fade at ROI borders (px)
    fill_holes=True,        # densify ROI
    close_kernel=15         # morph closing kernel (px)
):
    """
    Smooth, monotonic direction change across ROI (no jitter).
    Returns disp (H,W,2) with [dy,dx] and the soft weight map used.
    """
    H, W = tissue_mask.shape

    # 1) Dense ROI inside tissue (so every pixel in region moves)
    roi = (roi_partial_mask.astype(np.uint8) & tissue_mask.astype(np.uint8))
    if fill_holes:
        k = max(3, int(close_kernel) | 1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((k,k), np.uint8))
        # fill interior holes by flood-filling from canvas border
        ff = roi.copy()
        cv2.floodFill(ff, np.zeros((H+2, W+2), np.uint8), (0,0), 255)
        holes = (ff == 0).astype(np.uint8)
        roi = np.maximum(roi, holes) & tissue_mask

    # 2) Soft feather so deformation fades at ROI boundary (no sudden stop)
    soft = cv2.GaussianBlur(roi.astype(np.float32), (0,0), feather_px)
    if soft.max() > 0:
        soft = (soft - soft.min()) / (soft.max() - soft.min() + 1e-6)
    soft *= tissue_mask.astype(np.float32)

    # 3) Build a smooth angle map that varies ONLY along one axis
    ys, xs = np.mgrid[0:H, 0:W]
    y0, y1 = np.where(roi)[0].min(initial=0), np.where(roi)[0].max(initial=H-1)
    x0, x1 = np.where(roi)[1].min(initial=0), np.where(roi)[1].max(initial=W-1)

    if axis == 'vertical':
        # normalize 0 at top of ROI, 1 at bottom
        denom = max(1, (y1 - y0))
        t = np.clip((ys - y0) / denom, 0, 1).astype(np.float32)
    else:  # 'horizontal'
        denom = max(1, (x1 - x0))
        t = np.clip((xs - x0) / denom, 0, 1).astype(np.float32)

    # Optional easing for extra smoothness (cosine ease)
    t = 0.5 - 0.5 * np.cos(np.pi * t)

    # angle in radians, interpolated from top->bottom (or left->right)
    ang = np.deg2rad(angle_top_deg) * (1.0 - t) + np.deg2rad(angle_bottom_deg) * t

    # 4) Rotate the base direction by that angle field
    dx0, dy0 = base_direction
    mag0 = np.hypot(dx0, dy0) + 1e-6
    u0x, u0y = dx0 / mag0, dy0 / mag0

    ca, sa = np.cos(ang), np.sin(ang)
    dir_x = u0x * ca - u0y * sa
    dir_y = u0x * sa + u0y * ca

    # magnitude profile (constant inside ROI unless you want to vary it)
    if magnitude_px is None:
        magnitude_px = mag0

    disp_x = soft * magnitude_px * dir_x
    disp_y = soft * magnitude_px * dir_y

    # zero outside tissue
    disp_x *= tissue_mask.astype(np.float32)
    disp_y *= tissue_mask.astype(np.float32)

    disp = np.stack([disp_y, disp_x], axis=-1).astype(np.float32)
    return disp, soft
