print("Import packages ...")
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
from torch.utils.data import Dataset
from pytorch_msssim import ssim
import json
from itertools import cycle

print("Helper functions ...")


import torch
import torch.nn as nn
import torch.nn.functional as F

def balanced_batch_loader(easy_ds, med_ds, hard_ds, batch_size=3):
    easy_loader = cycle(DataLoader(easy_ds, batch_size=1, shuffle=True))
    med_loader = cycle(DataLoader(med_ds, batch_size=1, shuffle=True))
    hard_loader = cycle(DataLoader(hard_ds, batch_size=1, shuffle=True))

    while True:
        e = next(easy_loader)
        m = next(med_loader)
        h = next(hard_loader)

        batch_fixed = torch.cat([e[0], m[0], h[0]], dim=0)
        batch_moving = torch.cat([e[1], m[1], h[1]], dim=0)
        yield batch_fixed, batch_moving

def center_crop_1024x1024(tensor):
    """
    Crops the center 1024x1024 patch from a tensor of shape (1, 1, 1280, 1280).
    """
    assert tensor.shape[-2:] == (1280, 1280), "Input must have shape (*, 1280, 1280)"
    
    start_h = (tensor.shape[-2] - 1024) // 2
    start_w = (tensor.shape[-1] - 1024) // 2

    return tensor[:, :, start_h:start_h + 1024, start_w:start_w + 1024]

def center_crop_512x512(tensor):
    """
    Crops the center 1024x1024 patch from a tensor of shape (1, 1, 1280, 1280).
    """
    assert tensor.shape[-2:] == (768, 768), "Input must have shape (*, 1280, 1280)"
    
    start_h = (tensor.shape[-2] - 512) // 2
    start_w = (tensor.shape[-1] - 512) // 2

    return tensor[:, :, start_h:start_h + 512, start_w:start_w + 512]

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

# class Encoder(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.enc1 = ConvBlock(in_ch, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.enc2 = ConvBlock(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.enc3 = ConvBlock(128, 256)

#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool1(x1))
#         x3 = self.enc3(self.pool2(x2))
#         return x1, x2, x3

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Downsampling layer

        self.enc2 = ConvBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # Downsampling layer

        self.enc3 = ConvBlock(128, 256)

    def forward(self, x):
        x1 = self.enc1(x)
        x1_d = self.down1(x1)

        x2 = self.enc2(x1_d)
        x2_d = self.down2(x2)

        x3 = self.enc3(x2_d)
        return x1, x2, x3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(384, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(192, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)  # Output 2 channels: (dy, dx)

    def center_crop(self, x, target_height, target_width):
        _, _, h, w = x.size()
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        return x[:, :, start_h:start_h + target_height, start_w:start_w + target_width]

    def forward(self, f1, f2, f3):
        up1 = self.up1(f3)
        cat1 = torch.cat([up1, f2], dim=1)
        dec1 = self.dec1(cat1)

        up2 = self.up2(dec1)
        cat2 = torch.cat([up2, f1], dim=1)
        dec2 = self.dec2(cat2)

        out = self.final(dec2)
        out = self.center_crop(out, 512, 512)
        return out

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



def ncc_loss(warped, target, eps=1e-5):
    mean_warped = warped.mean(dim=[2, 3], keepdim=True)
    mean_target = target.mean(dim=[2, 3], keepdim=True)
    warped_centered = warped - mean_warped
    target_centered = target - mean_target
    numerator = (warped_centered * target_centered).mean(dim=[2, 3])
    denominator = (warped_centered.pow(2).mean(dim=[2, 3]) * target_centered.pow(2).mean(dim=[2, 3])).sqrt() + eps
    ncc = numerator / denominator
    return -ncc.mean()


def curvature_loss(flow):
    """
    Penalize curvature (second-order derivatives) of flow.
    flow: (B, 2, H, W)
    """
    lap_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32, device=flow.device).unsqueeze(0).unsqueeze(0)

    dx = flow[:, 1:2, :, :]  # (B, 1, H, W)
    dy = flow[:, 0:1, :, :]

    lap_dx = F.conv2d(dx, lap_kernel, padding=1)
    lap_dy = F.conv2d(dy, lap_kernel, padding=1)

    return (lap_dx.pow(2).mean() + lap_dy.pow(2).mean())

def custom_loss(pred, img1, img2, lambda_curv=0.5):
    """
    pred: predicted deformation field from model, shape (B, 2, H, W)
    img1: fixed image (torch.Tensor), shape (B, C, H, W)
    img2: moving image (torch.Tensor), shape (B, C, H, W)
    """
    # Make sure everything is on the same device
    device = pred.device
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Convert flow to grid (B, H, W, 2)
    flow = pred.permute(0, 1, 2, 3)

    # Warp the moving image toward the fixed image
    warped_img2 = apply_deformation_torch(img2, flow)
    # with torch.no_grad():
    #     baseline_ncc = ncc_loss(img2, img1)
    ncc = ncc_loss(warped_img2, img1)

    # compute curvature loss on (B, 2, H, W)
    curv = curvature_loss(pred)

    # Return NCC loss between warped and fixed
    loss = ncc + lambda_curv * curvature_loss(pred)
    # total_loss = ncc + lambda_curv * curv - ncc_loss(img2, img1)
    return loss

# def custom_loss(pred, img1, img2, lambda_curv=0.2, lambda_ssim=1.0):
#     """
#     pred: predicted deformation field from model, shape (B, 2, H, W)
#     img1: fixed image (torch.Tensor), shape (B, C, H, W)
#     img2: moving image (torch.Tensor), shape (B, C, H, W)
#     """
#     device = pred.device
#     img1 = img1.to(device)
#     img2 = img2.to(device)

#     # Convert flow to grid (B, H, W, 2)
#     flow = pred.permute(0, 1, 2, 3)

#     # Warp the moving image toward the fixed image
#     warped_img2 = apply_deformation_torch(img2, flow)

#     # Compute SSIM (higher is better, so we use 1 - SSIM as loss)
#     ssim_loss = 1 - ssim(warped_img2, img1, data_range=1.0, size_average=True)

#     # Compute curvature loss
#     curv = curvature_loss(pred)

#     # Final loss
#     total_loss = lambda_ssim * ssim_loss + lambda_curv * curv
#     return total_loss


class TensorPairDataset(Dataset):
    def __init__(self, fixed_list, moving_list):
        assert len(fixed_list) == len(moving_list)
        self.fixed_list = fixed_list
        self.moving_list = moving_list

    def __len__(self):
        return len(self.fixed_list)

    def __getitem__(self, idx):
        return self.fixed_list[idx], self.moving_list[idx]


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
    print(base_grid.shape, norm_deform.shape)
    grid = base_grid + norm_deform  # (B, H, W, 2)

    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)



def read_input_images_local(input_path: str):
    """
    Reads fixed/moving image patches from one input folder and groups them by NCC score.

    Args:
        input_path (str): Path to the main folder containing training subfolders (each with preprocess_out).
    Returns:
        buckets, fixed_buckets, moving_buckets, bucket_dataloaders, train_loader
    """

    print(f"Start loading patches from {input_path} ...")

    # Buckets by NCC score
    buckets = {"high": [], "medium": [], "low": []}
    fixed_buckets = {"high": [], "medium": [], "low": []}
    moving_buckets = {"high": [], "medium": [], "low": []}

    # Collect all subfolders
    subfolders = sorted([
        p for p in os.listdir(input_path)
        if p.split("_")[-1].isdigit() and 0 <= int(p.split("_")[-1]) <= 120
    ])

    for f in subfolders:
        print(f"Processing {f}...")
        fixed_dir = os.path.join(input_path, f, "patches_512_registered3", "fixed")
        moving_dir = os.path.join(input_path, f, "patches_512_registered3", "moving")

        if not os.path.exists(fixed_dir) or not os.path.exists(moving_dir):
            print(f"Missing fixed/moving folder for {f}, skipping.")
            continue

        fixed_files = sorted([x for x in os.listdir(fixed_dir) if x.startswith("patch")])
        moving_files = sorted([x for x in os.listdir(moving_dir) if x.startswith("patch")])

        for i in range(len(fixed_files)):
            fixed_path = os.path.join(fixed_dir, fixed_files[i])
            moving_path = os.path.join(moving_dir, moving_files[i])

            if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
                print(f"Missing files for {f}, skipping.")
                continue

            # Load image patches
            moving_patch = tiff.imread(moving_path).astype(float)
            fixed_patch = tiff.imread(fixed_path).astype(float).mean(axis=2)

            # Normalize
            moving_patch_bw = moving_patch / 255.0 if moving_patch.max() > 1 else moving_patch
            fixed_patch_bw = fixed_patch / 255.0 if fixed_patch.max() > 1 else fixed_patch

            # Convert to tensors
            fixed_tensor = torch.from_numpy(fixed_patch_bw).float().unsqueeze(0).unsqueeze(0)
            moving_tensor = torch.from_numpy(moving_patch_bw).float().unsqueeze(0).unsqueeze(0)

            # Compute NCC score
            with torch.no_grad():
                score = ncc_loss(fixed_tensor, moving_tensor).item()

            # Bucket assignment
            if score > -0.2:
                bucket = "high"
            elif score > -0.4:
                bucket = "medium"
            else:
                bucket = "low"

            # NaN safety check
            if torch.isnan(fixed_tensor).any() or torch.isnan(moving_tensor).any():
                print("Skipping NaN patch.")
                continue

            buckets[bucket].append((fixed_tensor.squeeze(0), moving_tensor.squeeze(0)))
            fixed_buckets[bucket].append(fixed_tensor.squeeze(0))
            moving_buckets[bucket].append(moving_tensor.squeeze(0))
            print(f"{f}, Patch {i}, NCC: {score:.4f}, Bucket: {bucket}")

    # Create bucket dataloaders
    print("Building dataset...")
    bucket_dataloaders = {}
    for bucket in ["high", "medium", "low"]:
        dataset = TensorPairDataset(fixed_buckets[bucket], moving_buckets[bucket])
        bucket_dataloaders[bucket] = DataLoader(dataset, batch_size=3, shuffle=True)

    # Balanced batch loader
    train_loader = balanced_batch_loader(
        bucket_dataloaders["low"].dataset,
        bucket_dataloaders["medium"].dataset,
        bucket_dataloaders["high"].dataset
    )

    return buckets, fixed_buckets, moving_buckets, bucket_dataloaders, train_loader


def train_local(save_path: str, train_loader):

    num_epochs = 300
    loss_history = []
    # bucket_dataloaders["high"], bucket_dataloaders["medium"], bucket_dataloaders["low"]
    num_batches = max(len(bucket_dataloaders["low"]), len(bucket_dataloaders["medium"]), len(bucket_dataloaders["high"]))
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}...")
        count = 0

        for i in range(num_batches):
            fixed, moving = next(train_loader)
            fixed, moving = fixed.to(device), moving.to(device)
            fixed_center = center_crop_512x512(fixed)
            moving_center = center_crop_512x512(moving)
            if torch.isnan(fixed).any():
                print("fixed contains NaNs!")
            if torch.isnan(moving).any():
                print("moving contains NaNs!")
            pred_flow = model(fixed, moving)
            if torch.isnan(pred_flow).any():
                print(f"{count} pred_flow contains NaNs!")
            loss = custom_loss(pred_flow, fixed_center, moving_center)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1} - Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / num_batches:.6f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'loss_history': loss_history
    }, save_path)
    print(f"Model saved to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Deformation Training")
    parser.add_argument("--input_path", required=True, help="Path to the folder with training subfolders")
    parser.add_argument("--save_path", required=True, help="Where to save the final checkpoint")
    args = parser.parse_args()

    buckets, fixed_buckets, moving_buckets, bucket_dataloaders, train_loader = read_input_images_local(args.input_path)
    train_local(save_path=args.save_path, train_loader=train_loader)

if __name__ == "__main__":
    main()
