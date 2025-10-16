import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import matplotlib.cm as cm

from utils.synthesis import canonical_2d, grids2disps


def get_env():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, local_rank, device


def save_ckpt(save_dir, epoch, model, comp_net=None, optimizer=None, scheduler=None, 
              loss=None, lpips=None, ssim=None, name=None):
    
    model_state_dict = model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict
    }
    
    if comp_net is not None:
        checkpoint["comp_net_state_dict"] = comp_net.state_dict()
        
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if loss is not None:
        checkpoint["loss"] = loss
        
    if lpips is not None:
        checkpoint["lpips"] = lpips
        
    if ssim is not None:
        checkpoint["ssim"] = ssim
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{epoch if name is None else name}.pth")
    torch.save(checkpoint, save_path)


def load_ckpt(checkpoint_path, model, comp_net=None, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if comp_net is not None:
        comp_net.load_state_dict(checkpoint["comp_net_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    
    metrics = {}
    if "loss" in checkpoint:
        metrics["loss"] = checkpoint["loss"]
    if "lpips" in checkpoint:
        metrics["lpips"] = checkpoint["lpips"]
    if "ssim" in checkpoint:
        metrics["ssim"] = checkpoint["ssim"]
    
    return epoch, metrics


def get_img(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32)


def get_npy(path):
    img = np.load(path).transpose(1, 2, 0)
    return img.astype(np.float32)


def visualize_blur_trajectories(disps, output_path=None, spacing=20, colormap=None):
    """
    Visualize blur trajectories from normalized grids with optional gradient coloring.
    
    Parameters:
    -----------
    grids : np.ndarray
        Normalized grids of shape [num_poses, H, W, 2] in [-1, 1] space
        Last dimension contains (x, y) coordinates in normalized space
    output_path : str
        Path to save the visualization
    spacing : int
        Spacing between reference points (default: 20)
    colormap : str or None
        Matplotlib colormap name for gradient coloring (default: None)
        If None, plots white dots on black background
    
    Returns:
    --------
    np.ndarray : The visualization image (RGB if colormap, grayscale if None)
    """
    num_poses, H, W, _ = disps.shape
    
    # Calculate number of grid points that will fit
    num_y = H // spacing
    num_x = W // spacing
    
    # Calculate offsets to center the grid
    offset_y = (H - (num_y - 1) * spacing) // 2
    offset_x = (W - (num_x - 1) * spacing) // 2
    
    # Create centered sparse grid of reference points
    y_coords = np.arange(num_y) * spacing + offset_y
    x_coords = np.arange(num_x) * spacing + offset_x
    
    # Create meshgrid for vectorized operations
    y_ref_grid, x_ref_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten to get all reference points as arrays
    y_ref_flat = y_ref_grid.flatten()
    x_ref_flat = x_ref_grid.flatten()
    
    # Centering
    disps -= np.mean(disps, axis=0, keepdims=True)
    
    # Decide on output type based on colormap
    if colormap is None:
        # Grayscale output with white dots
        output_image = np.zeros((H, W), dtype=np.uint8)
        use_color = False
    else:
        # RGB output with gradient colors
        output_image = np.zeros((H, W, 3), dtype=np.uint8)
        cmap = cm.get_cmap(colormap)
        use_color = True
    
    # For each pose, assign a color based on its position in the sequence
    for pose_idx in range(num_poses):
        if use_color:
            # Normalize pose index to [0, 1] for colormap
            t = pose_idx / max(1, num_poses - 1)
            
            # Get RGB color from colormap (values in [0, 1])
            color_normalized = cmap(t)[:3]  # Take only RGB, ignore alpha
            color = (np.array(color_normalized) * 255).astype(np.uint8)
        else:
            # White color for grayscale
            color = 255
        
        # Extract displacements for all reference points at this pose
        dy = disps[pose_idx, y_ref_flat, x_ref_flat, 1]
        dx = disps[pose_idx, y_ref_flat, x_ref_flat, 0]
        
        # Calculate warped positions (vectorized)
        y_warped = np.round(y_ref_flat + dy).astype(int)
        x_warped = np.round(x_ref_flat + dx).astype(int)
        
        # Filter valid coordinates (within bounds)
        valid_mask = (y_warped >= 0) & (y_warped < H) & (x_warped >= 0) & (x_warped < W)
        y_valid = y_warped[valid_mask]
        x_valid = x_warped[valid_mask]
        
        # Paint pixels
        output_image[y_valid, x_valid] = color
    
    # Save the image
    if output_path:
        if use_color:
            Image.fromarray(output_image, mode='RGB').save(output_path)
        else:
            Image.fromarray(output_image, mode='L').save(output_path)
    
    return output_image
