import os
import torch
import numpy as np
import tifffile as tiff
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
from scipy import stats
import argparse

# --- Helper Functions ---

def replace_black_with_mode_background(image):
    """
    Replace black (zero) pixels with the mode of nonzero pixels.
    Works best when background dominates the image.
    """
    nonzero_pixels = image[image > 0]
    if len(nonzero_pixels) == 0:
        return image
    mode_val = stats.mode(nonzero_pixels, axis=None, keepdims=False).mode
    image_filled = image.copy()
    image_filled[image == 0] = mode_val
    return image_filled

def create_padding_mask(h, w, target_h=1024, target_w=1024):
    mask = torch.zeros((target_h, target_w), dtype=torch.float32)
    start_h = (target_h - h) // 2
    start_w = (target_w - w) // 2
    mask[start_h:start_h + h, start_w:start_w + w] = 1.0
    return mask

def center_crop(tensor, crop_h, crop_w):
    """
    Crops the center (crop_h, crop_w) region from a (B, C, H, W) tensor.
    """
    crop_h = int(crop_h)
    crop_w = int(crop_w)
    _, _, H, W = tensor.shape
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2
    return tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

def upscale_deformation_field(flow, new_size):
    """
    Upscales a backward deformation field (B, 2, H, W) to new_size (H_new, W_new),
    and scales displacement vectors accordingly.
    """
    B, C, H, W = flow.shape
    new_H, new_W = new_size

    scale_y = new_H / H
    scale_x = new_W / W

    # Upsample spatially
    flow_up = F.interpolate(flow, size=(new_H, new_W), mode='bilinear', align_corners=True)

    # Scale displacement values
    flow_up[:, 0] *= scale_y  # dy
    flow_up[:, 1] *= scale_x  # dx

    return flow_up

def pad_to_1024(image):
    h, w = image.shape[:2]
    pad_h = max(0, 1024 - h)
    pad_w = max(0, 1024 - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    return np.pad(image, padding, mode='constant', constant_values=0)

def apply_deformation_torch(image, deformation):
    B, C, H, W = image.shape

    # Create base normalized grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing='ij'
    )
    base_grid = torch.stack((x, y), dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

    # Normalize deformation to [-1, 1] range
    norm_deform = torch.stack([
        2.0 * deformation[:, 1, ...] / (W - 1),
        2.0 * deformation[:, 0, ...] / (H - 1)
    ], dim=-1)
    
    grid = base_grid + norm_deform  # (B, H, W, 2)

    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)

# --- Model Definition ---

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        return x1, x2, x3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(384, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(192, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)  # Output 2 channels: (dy, dx)

    def forward(self, f1, f2, f3):
        up1 = self.up1(f3)
        cat1 = torch.cat([up1, f2], dim=1)
        dec1 = self.dec1(cat1)

        up2 = self.up2(dec1)
        cat2 = torch.cat([up2, f1], dim=1)
        dec2 = self.dec2(cat2)

        return self.final(dec2)

class DualEncoderUNet(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.encoder1 = Encoder(in_ch)
        self.encoder2 = Encoder(in_ch)
        self.decoder = Decoder()

    def forward(self, img1, img2):
        f11, f12, f13 = self.encoder1(img1)
        f21, f22, f23 = self.encoder2(img2)

        f1 = torch.cat([f11, f21], dim=1)
        f2 = torch.cat([f12, f22], dim=1)
        f3 = torch.cat([f13, f23], dim=1)

        return self.decoder(f1, f2, f3)

# --- Main Inference Loop ---

def run_inference(input_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DualEncoderUNet(in_ch=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Get list of subdirectories
    path_list = sorted([
        p for p in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, p)) and p.split('_')[-1].isdigit()
    ])

    if not path_list:
        print("No valid subdirectories found.")
        return

    for f in path_list:
        print(f"Processing {f}...")
        current_dir = os.path.join(input_path, f)
        fixed_path = os.path.join(current_dir, "preprocess_out/cropped_fixed.tif")
        moving_path = os.path.join(current_dir, "preprocess_out/cropped_moving.tif")
        output_path = os.path.join(current_dir, "preprocess_out/global_registered.tif")

        if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
            print(f"Missing files for {f}, skipping.")
            continue

        try:
            # Read images
            fixed = tiff.imread(fixed_path)
            moving = tiff.imread(moving_path)

            # Handle multi-channel images (take mean if RGB)
            if fixed.ndim == 3:
                fixed = fixed.mean(axis=2)
            if moving.ndim == 3:
                moving = moving.mean(axis=2)

            # Preprocess moving image
            moving_processed = replace_black_with_mode_background(moving)

            # Resize
            factor = 1 / math.ceil(max(fixed.shape) / 1024)
            temp_fixed = cv2.resize(fixed, None, fx=factor, fy=factor)
            temp_moving = cv2.resize(moving_processed, None, fx=factor, fy=factor)

            # Normalize
            temp_moving = temp_moving / 255.0 if temp_moving.max() > 1 else temp_moving
            temp_fixed = temp_fixed / 255.0 if temp_fixed.max() > 1 else temp_fixed

            # Pad to 1024
            fixed_resized = pad_to_1024(temp_fixed)
            moving_resized = pad_to_1024(temp_moving)

            # Convert to tensor
            fixed_tensor = torch.from_numpy(fixed_resized).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)
            moving_tensor = torch.from_numpy(moving_resized).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)

            # Inference
            with torch.no_grad():
                output = model(fixed_tensor, moving_tensor)

            # Crop output to match temp_fixed size (remove padding)
            cropped = center_crop(output, temp_fixed.shape[0], temp_fixed.shape[1])

            # Upscale deformation field to original moving image size
            highres_flow = upscale_deformation_field(cropped, new_size=(moving.shape[0], moving.shape[1]))

            # Warp original moving image
            image_moving_gray_torch = torch.from_numpy(moving).float().unsqueeze(0).unsqueeze(0).to(device)
            warped_high_res = apply_deformation_torch(image_moving_gray_torch, highres_flow)

            # Save result
            result_numpy = warped_high_res.detach().cpu().squeeze(0).squeeze(0).numpy()
            tiff.imwrite(output_path, result_numpy.astype(np.float32))
            print(f"Saved result to {output_path}")

        except Exception as e:
            print(f"Error processing {f}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run global deformation inference")
    parser.add_argument("--input_dir", required=True, help="Path to the dataset directory containing subfolders")
    parser.add_argument("--model_path", required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()

    run_inference(args.input_dir, args.model_path)

if __name__ == "__main__":
    main()
