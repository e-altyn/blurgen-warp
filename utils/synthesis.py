# based on snippet from https://openreview.net/pdf?id=Wvi8c0tgvt

import math

from functools import cache

import torch
import torch.nn.functional as F


def vec2skew(v):
    """Skew-symmetric matrix for batch of vectors.
    
    Args:
        v: [N, 3] batch of 3D vectors
        
    Returns:
        skew_v: [N, 3, 3] batch of skew-symmetric matrices
    """
    N = v.shape[0]
    device = v.device
    zero = torch.zeros(N, 1, dtype=torch.float32, device=device)
    
    # Extract components
    v0, v1, v2 = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    
    # Build rows of skew-symmetric matrix
    skew_v0 = torch.cat([zero, -v2, v1], dim=1)  # [N, 3]
    skew_v1 = torch.cat([v2, zero, -v0], dim=1)  # [N, 3]
    skew_v2 = torch.cat([-v1, v0, zero], dim=1)  # [N, 3]
    
    # Stack into matrix
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # [N, 3, 3]
    
    return skew_v


def batch_Exp(r):
    """Rodrigues' formula for batch of rotation vectors.
    
    Args:
        r: [N, 3] batch of rotation vectors (axis-angle representation)
        
    Returns:
        R: [N, 3, 3] batch of rotation matrices
    """
    N = r.shape[0]
    device = r.device
    
    # Compute skew-symmetric matrices in batch
    skew_r = vec2skew(r)  # [N, 3, 3]
    
    # Compute norms [N, 1, 1] for broadcasting
    norm_r = r.norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-15
    
    # Identity matrices [N, 3, 3]
    eye = torch.eye(3, dtype=torch.float32, device=device)[None, :, :].expand(N, -1, -1)
    
    # Rodrigues formula (vectorized)
    sin_term = torch.sin(norm_r) / norm_r
    cos_term = (1 - torch.cos(norm_r)) / (norm_r ** 2)
    
    R = eye + sin_term * skew_r + cos_term * (skew_r @ skew_r)
    
    return R


def make_c2w(r, t):
    """Camera-to-world matrix construction.
    
    Args:
        r: [N, 3] batch of rotation parameters
        t: [N, 3] batch of translation parameters
        
    Returns:
        c2w: [N, 3, 4] batch of camera-to-world matrices
    """
    R = batch_Exp(r)  # [N, 3, 3]
    t = t.unsqueeze(-1)  # [N, 3, 1]
    c2w = torch.cat([R, t], dim=-1)  # [N, 3, 4]
    return c2w


def complex_to_polar(z):
    """Convert complex representation to polar coordinates."""
    z_complex = z[..., 0] + 1j * z[..., 1]
    return torch.abs(z_complex), torch.angle(z_complex)


def polar_to_complex(amplitude, phase):
    """Convert polar coordinates to complex representation."""
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    return torch.stack((real, imag), dim=-1)

@cache
def canonical_2d(H, W, device='cpu', in_pixel_space=False):
    """Returns [1, H, W, 2] canionical grid."""
    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    grid = F.affine_grid(identity, [1, 3, H, W], 
                         align_corners=False).to(device)
    if in_pixel_space:
        return grid2pixel(grid, H, W, align_corners=False)
    else:
        return grid


def vec_field_gen(motion_results, control_params, img_size):
    """Vector field generation.
    
    Args:
        motion_results: tuple of (r, t, amp_local, phase_local)
            - r: [B, num_poses * 3] rotation parameters
            - t: [B, num_poses * 3] translation parameters
            - amp_local: [B, num_poses, h, 2, w] predicted in-frame displacements
            - phase_local: [B, num_poses, h, w] local phase
        control_params: tuple of (amp_control, phase_control)
            - amp_control: scalar or broadcastable tensor
            - phase_control: scalar or broadcastable tensor
        img_size: torch.Size([B, c, H, W])
        
    Returns:
        total_grid: [B, num_poses, H, W, 2]
        total_grid_invert: [B, num_poses, H, W, 2]
        displacements : [B, num_poses, H, W, 2] in pixel space (model predictions)
    """
    r, t, amp_local, phase_local = motion_results
    amp_control, phase_control = control_params
    
    B, _, H, W = img_size
    num_poses = amp_local.shape[1]
    
    # Reshape r and t: [B, num_poses * 3] -> [B, num_poses, 3] -> [B * num_poses, 3]
    r = r.reshape(B, num_poses, 3).reshape(B * num_poses, 3)
    t = t.reshape(B, num_poses, 3).reshape(B * num_poses, 3)
    
    # Batch compute all c2w matrices at once
    rigid_2d = make_c2w(r, t)  # [B * num_poses, 3, 4]
    
    # Get canonical grid and move to device
    grid_2d_cano = canonical_2d(H, W).to(r.device)  # [1, H, W, 2]
    
    # Compute rigid transform for all poses at once
    grid_2d_rigid = F.affine_grid(rigid_2d[:, :2, :3], [B * num_poses, img_size[1], H, W])  # [B * num_poses, H, W, 2]
    
    # Compute residual
    grid_2d_cano_expanded = grid_2d_cano.expand(B * num_poses, -1, -1, -1)
    res_grid_2d_rigid = grid_2d_rigid - grid_2d_cano_expanded[..., :2]  # [B * num_poses, H, W, 2]
    
    # Convert to polar coordinates
    amp_global, phase_global = complex_to_polar(res_grid_2d_rigid)  # [B * num_poses, H, W]
    
    # Reshape back to [B, num_poses, H, W] for element-wise operations
    amp_global = amp_global.reshape(B, num_poses, H, W)
    phase_global = phase_global.reshape(B, num_poses, H, W)
    
    # Apply local and control parameters (fully vectorized)
    res_grid_3d_aware = polar_to_complex(
        amp_global * amp_local * amp_control,
        phase_global + phase_local + phase_control,
    )  # [B, num_poses, H, W, 2]
    
    # Add canonical grid using broadcasting
    grid_2d_cano_broadcast = grid_2d_cano[None, ...].expand(B, num_poses, -1, -1, -1)
    total_grid = res_grid_3d_aware + grid_2d_cano_broadcast
    total_grid_invert = -res_grid_3d_aware + grid_2d_cano_broadcast
    
    return total_grid, total_grid_invert


def blur_synthesis(model, blur_img, sharp_img, control_params=(1, 0), 
                   num_poses=16, compensation_net=None):
    # Generate grids
    grids, grids_inv = vec_field_gen(
        model(blur_img), control_params, sharp_img.size()
    )
    
    B, C, H, W = sharp_img.shape
    
    # Expand sharp_img: [B, C, H, W] -> [B*num_poses, C, H, W]
    sharp_img_expanded = sharp_img.unsqueeze(1).expand(B, num_poses, C, H, W)
    sharp_img_batched = sharp_img_expanded.reshape(B * num_poses, C, H, W)
    
    # Reshape grids: [B, num_poses, H, W, 2] -> [B*num_poses, H, W, 2]
    grids_batched = grids.reshape(B * num_poses, H, W, 2)
    grids_inv_batched = grids_inv.reshape(B * num_poses, H, W, 2)
    
    # Vectorized warping
    warped = F.grid_sample(sharp_img_batched, grids_batched)
    
    # Add noise
    gaussian_noise = torch.randn_like(warped, requires_grad=False) * 0.0112
    warped = torch.clamp(warped + gaussian_noise, -1, 1)
    
    # Cycle warping
    cycle_warped = F.grid_sample(warped, grids_inv_batched)
    
    # Reshape back: [B*num_poses, C, H, W] -> [B, num_poses, C, H, W]
    warped = warped.reshape(B, num_poses, C, H, W)
    cycle_warped = cycle_warped.reshape(B, num_poses, C, H, W)
    
    # Average across poses
    warped_mean = warped.mean(dim=1)
    
    results = {
        'pred_blur': warped_mean,
        'cycle_warped_img': cycle_warped,
        'grids': grids,
        'grids_inv': grids_inv
    }
    
    if compensation_net is not None:
        results['pred_blur_comp'] = compensation_net(warped_mean)
    
    return results


@torch.no_grad
def blur_data_augmentation(blur_img, sharp_img, model):
    B = blur_img.size()[0]
    control_params = (
        torch.empty(B, 1, 1).uniform_(1.0, 2.0),
        torch.empty(B, 1, 1).uniform_(-1.0, 1.0) * (math.pi / 2.0)
    )

    target_shuf = sharp_img[torch.randperm(B)]
    results = blur_synthesis(model, blur_img, target_shuf, control_params)
    
    return results['pred_blur']


def grid2pixel(grid, H, W, align_corners=False):
    """
    Convert normalized grid coordinates to pixel coordinates.
    
    Args:
        grid: Tensor of shape (..., 2) with normalized coordinates in range [-1, 1]
              where [..., 0] is x (width) and [..., 1] is y (height)
        H: Image height in pixels
        W: Image width in pixels
        align_corners: If True, uses corner pixel centers; if False, uses pixel edges
        
    Returns:
        Tensor of same shape with pixel coordinates
    """
    pixel_coords = grid.clone()
    
    if align_corners:
        # Map [-1, 1] to [0, W-1] for x and [0, H-1] for y
        pixel_coords[..., 0] = (grid[..., 0] + 1) * (W - 1) / 2
        pixel_coords[..., 1] = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        # Map [-1, 1] to [0, W] for x and [0, H] for y
        pixel_coords[..., 0] = (grid[..., 0] + 1) * W / 2
        pixel_coords[..., 1] = (grid[..., 1] + 1) * H / 2
    
    return pixel_coords


def pixel2grid(pixel_coords, H, W, align_corners=False):
    """
    Convert pixel coordinates to normalized grid coordinates.
    
    Args:
        pixel_coords: Tensor of shape (..., 2) with pixel coordinates
                      where [..., 0] is x (width) and [..., 1] is y (height)
        H: Image height in pixels
        W: Image width in pixels
        align_corners: If True, uses corner pixel centers; if False, uses pixel edges
        
    Returns:
        Tensor of same shape with normalized coordinates in range [-1, 1]
    """
    grid = pixel_coords.clone()
    
    if align_corners:
        # Map [0, W-1] to [-1, 1] for x and [0, H-1] to [-1, 1] for y
        grid[..., 0] = 2 * pixel_coords[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * pixel_coords[..., 1] / (H - 1) - 1
    else:
        # Map [0, W] to [-1, 1] for x and [0, H] to [-1, 1] for y
        grid[..., 0] = 2 * pixel_coords[..., 0] / W - 1
        grid[..., 1] = 2 * pixel_coords[..., 1] / H - 1
    
    return grid
