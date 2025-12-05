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
import json
from pytorch_msssim import ssim
from scipy import stats

print("Helper functions ...")

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

def create_padding_mask(h, w, target_h=1024, target_w=1024):
    mask = torch.zeros((target_h, target_w), dtype=torch.float32)
    start_h = (target_h - h) // 2
    start_w = (target_w - w) // 2
    mask[start_h:start_h + h, start_w:start_w + w] = 1.0
    return mask

def center_crop(tensor, crop_h, crop_w):
    """
    Crops the center (704, 1006) region from a (B, C, H, W) tensor.
    """
    crop_h = int(crop_h)
    crop_w = int(crop_w)
    _, _, H, W = tensor.shape
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2
    return tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

def center_crop_1024x1024(tensor):
    """
    Crops the center 1024x1024 patch from a tensor of shape (1, 1, 1280, 1280).
    """
    assert tensor.shape[-2:] == (1280, 1280), "Input must have shape (*, 1280, 1280)"
    
    start_h = (tensor.shape[-2] - 1024) // 2
    start_w = (tensor.shape[-1] - 1024) // 2

    return tensor[:, :, start_h:start_h + 1024, start_w:start_w + 1024]


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



def ncc_loss(warped, target, eps=1e-5):
    mean_warped = warped.mean(dim=[2, 3], keepdim=True)
    mean_target = target.mean(dim=[2, 3], keepdim=True)
    warped_centered = warped - mean_warped
    target_centered = target - mean_target
    numerator = (warped_centered * target_centered).mean(dim=[2, 3])
    denominator = (warped_centered.pow(2).mean(dim=[2, 3]) * target_centered.pow(2).mean(dim=[2, 3])).sqrt() + eps
    ncc = numerator / denominator
    return -ncc.mean()

def masked_ncc_loss(warped, target, mask, eps=1e-5):
    masked_warped = warped * mask
    masked_target = target * mask

    mean_warped = masked_warped.sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[1, 2], keepdim=True) + eps)
    mean_target = masked_target.sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[1, 2], keepdim=True) + eps)

    warped_centered = masked_warped - mean_warped
    target_centered = masked_target - mean_target

    numerator = (warped_centered * target_centered * mask).sum(dim=[2, 3])
    denominator = (
        (warped_centered.pow(2) * mask).sum(dim=[2, 3]) *
        (target_centered.pow(2) * mask).sum(dim=[2, 3])
    ).sqrt() + eps

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

def custom_loss(pred, img1, img2, padding_mask, lambda_curv=0.2):
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
    # ncc = ncc_loss(warped_img2, img1)
    ncc = masked_ncc_loss(warped_img2, img1, padding_mask)

    # compute curvature loss on (B, 2, H, W)
    curv = curvature_loss(pred)

    # Return NCC loss between warped and fixed
    # total_loss = ncc + lambda_curv * curv - ncc_loss(img2, img1)
    total_loss = ncc + lambda_curv * curv
    return total_loss

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
    def __init__(self, fixed_list, moving_list, orig_size_list):
        assert len(fixed_list) == len(moving_list)
        self.fixed_list = fixed_list
        self.moving_list = moving_list
        self.orig_size_list = orig_size_list

    def __len__(self):
        return len(self.fixed_list)

    def __getitem__(self, idx):
        return self.fixed_list[idx], self.moving_list[idx], self.orig_size_list[idx]


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


# def apply_deformation_torch(image, deformation):
#     B, C, H, W = image.shape
#     grid = torch.stack([
#         2.0 * deformation[..., 1] / (W - 1) - 1.0,
#         2.0 * deformation[..., 0] / (H - 1) - 1.0
#     ], dim=-1)
#     return F.grid_sample(image, grid, mode='bilinear', align_corners=True)

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

def load_training_tensors(input_path: str):
    """
    Load and preprocess fixed/moving training image pairs from a single folder.
    Args:
        input_path (str): Path to the folder containing training subfolders.
    Returns:
        fixed_list (list[torch.Tensor])
        moving_list (list[torch.Tensor])
        original_size (list[np.ndarray or torch.Tensor])
    """
    print('Start loading training data ...')

    fixed_list = []
    moving_list = []
    original_size = []

    # Gather training subfolder names
    path_list = sorted([
        p for p in os.listdir(input_path)
        if p.split('_')[-1].isdigit() and 0 <= int(p.split('_')[-1]) <= 120
    ])

    for f in path_list:
        print(f"Processing {f}...")
        fixed_path = f"{input_path}/{f}/preprocess_out/cropped_fixed.tif"
        moving_path = f"{input_path}/{f}/preprocess_out/cropped_moving.tif"

        if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
            print(f"Missing files for {f}, skipping.")
            continue

        # Read and preprocess
        fixed = tiff.imread(fixed_path).mean(axis=2)
        moving = tiff.imread(moving_path).mean(axis=2)

        # Resize proportionally to max 1024
        factor = 1 / math.ceil(max(fixed.shape) / 1024)
        temp_fixed = cv2.resize(fixed, None, fx=factor, fy=factor)
        temp_moving = cv2.resize(moving, None, fx=factor, fy=factor)
        temp_moving = replace_black_with_mode_background(temp_moving)

        temp_moving = temp_moving / 255.0 if temp_moving.max() > 1 else temp_moving
        temp_fixed = temp_fixed / 255.0 if temp_fixed.max() > 1 else temp_fixed

        temp1 = torch.from_numpy(temp_fixed).float().unsqueeze(0)
        temp2 = torch.from_numpy(temp_moving).float().unsqueeze(0)

        # Quality checks
        if float(ncc_loss(torch.unsqueeze(temp1, 0), torch.unsqueeze(temp2, 0))) > 0:
            print('Bad pair')
            continue
        elif torch.isnan(temp1).any() or torch.isnan(temp2).any():
            print("Pair contain NaNs!")
            continue

        # Pad to 1024×1024
        fixed_resized = pad_to_1024(temp_fixed)
        moving_resized = pad_to_1024(temp_moving)
        fixed_tensor = torch.from_numpy(fixed_resized).float().unsqueeze(0)
        moving_tensor = torch.from_numpy(moving_resized).float().unsqueeze(0)
        fixed_list.append(fixed_tensor)
        moving_list.append(moving_tensor)

        # Record original size as padding mask
        padding_mask = create_padding_mask(temp_fixed.shape[0], temp_fixed.shape[1])
        original_size.append(padding_mask)

    return fixed_list, moving_list, original_size









def train_model(input_path, save_path):
    print("Building dataset...")
    fixed_list, moving_list, original_size = load_training_tensors(input_path)
    dataset = TensorPairDataset(fixed_list, moving_list, original_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderUNet(in_ch=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}...")
        count = 0
        for i, (fixed, moving, orig_size) in enumerate(dataloader):
            fixed, moving, orig_size= fixed.to(device), moving.to(device), orig_size.to(device)
            if torch.isnan(fixed).any():
                print("fixed contains NaNs!")
            if torch.isnan(moving).any():
                print("moving contains NaNs!")
            pred_flow = model(fixed, moving)
            if torch.isnan(pred_flow).any():
                print(f"{count} pred_flow contains NaNs!")
            # print('sadf: ', orig_size)

            # moving_cropped = center_crop(moving, orig_size[0], orig_size[1])
            # fixed_cropped = center_crop(fixed, orig_size[0], orig_size[1])
            # moving_cropped = center_crop(moving, orig_w, orig_h)
            # fixed_cropped = center_crop(fixed, orig_w, orig_h)
            # # break
            # pred_flow_cropped = center_crop(pred_flow, orig_w, orig_h)
        

        
            # loss = custom_loss(pred_flow_cropped, fixed_cropped, moving_cropped, padding_mask=orig_size)
            loss = custom_loss(pred_flow, fixed, moving, padding_mask=orig_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1} - Loss: {loss.item():.6f}")
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(dataloader):.6f}")

    # save_path = "/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_real_sample_v3.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, save_path)
    print(f"Model saved to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run global deformation training")
    parser.add_argument("--input_path", required=True, help="Where the training images are")
    parser.add_argument("--save_path", required=True, help="Where to save the final checkpoint")
    args = parser.parse_args()
    train_model(args.input_path, args.save_path)

if __name__ == "__main__":
    main()
