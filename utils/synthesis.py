# based on snippet from https://openreview.net/pdf?id=Wvi8c0tgvt

import math

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


def canonical_2d(H, W, device='cpu', normalized=True):
    """Returns [1, H, W, 2] canonical grid.
    
    Assumes align_corners = False.
    Args:
        H: Image height in pixels
        W: Image width in pixels
        device: Device to create tensors on
        normalized: If True, returns normalized [-1, 1] coordinates; if False, returns pixel ones
    
    Returns:
        Tensor of shape [1, H, W, 2] with grid coordinates
    """
    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device)
    grid = F.affine_grid(identity, [1, 3, H, W], align_corners=False)
    
    if normalized:
        return grid
    else:
        pixel_grid = torch.empty_like(grid)
        pixel_grid[..., 0] = (grid[..., 0] + 1) * W / 2 - 0.5
        pixel_grid[..., 1] = (grid[..., 1] + 1) * H / 2 - 0.5
        return pixel_grid


def denormalize_fields(grids, align_corners=False):
    """
    Convert normalized grid coordinates to pixel coordinates. 
    
    Args:
        grids: Tensor of shape [..., H, W, 2] with normalized coordinates in range [-1, 1]
               where [..., 0] is x (width) and [..., 1] is y (height)
        align_corners: If True, uses corner pixel centers; if False, uses pixel edges
    
    Returns:
        Tensor of same shape with pixel coordinates
    """
    H, W = grids.shape[-3:-1]
    pixel_coords = torch.empty_like(grids)
    
    if align_corners:
        # Map [-1, 1] to [0, W-1] for x and [0, H-1] for y
        pixel_coords[..., 0] = (grids[..., 0] + 1) * (W - 1) / 2
        pixel_coords[..., 1] = (grids[..., 1] + 1) * (H - 1) / 2
    else:
        # Map [-1, 1] to [0, W] for x and [0, H] for y
        pixel_coords[..., 0] = (grids[..., 0] + 1) * W / 2 - 0.5
        pixel_coords[..., 1] = (grids[..., 1] + 1) * H / 2 - 0.5

    
    return pixel_coords


def normalize_fields(disps, align_corners=False):
    """
    Convert pixel coordinates to normalized grid coordinates.
    
    Args:
        disps: Tensor of shape [..., H, W, 2] with pixel coordinates
               where [..., 0] is x (width) and [..., 1] is y (height)
        align_corners: If True, uses corner pixel centers; if False, uses pixel edges
    
    Returns:
        Tensor of same shape with normalized coordinates in range [-1, 1]
    """
    H, W = disps.shape[-3:-1]
    grids = torch.empty_like(disps)
    
    if align_corners:
        # Map [0, W-1] to [-1, 1] for x and [0, H-1] to [-1, 1] for y
        grids[..., 0] = 2 * disps[..., 0] / (W - 1) - 1
        grids[..., 1] = 2 * disps[..., 1] / (H - 1) - 1
    else:
        # Map [0, W] to [-1, 1] for x and [0, H] to [-1, 1] for y
        grids[..., 0] = 2 * (disps[..., 0] + 0.5) / W - 1
        grids[..., 1] = 2 * (disps[..., 1] + 0.5) / H - 1

    return grids


def scale_rotate(model_output, control_params):
    B, num_poses, H, W, _ = model_output.shape
    amp_control, phase_control = control_params
    
    amp, phase = complex_to_polar(model_output)
    amp = amp.reshape(B, num_poses, H, W)
    phase = phase.reshape(B, num_poses, H, W)
    
    final_disps = polar_to_complex(
        amp * amp_control,
        phase + phase_control,
    )  # [B, num_poses, H, W, 2]

    return final_disps


def grid_sample(img, disps):
    """Grid sample along displacement fields.
    
    Args:
        img: [N, 3, H, W]
        disps: Normalized to [-1, 1] displacemets [N, H, W, 2]
    Returns:
        warped: [N, 3, H, W]
    """
    N, H, W, _ = disps.shape
    #grids = normalize_fields(disps + canonical_2d(H, W, device=disps.device), align_corners=False)
    grids = disps + canonical_2d(H, W, device=disps.device, normalized=True)
    
    return F.grid_sample(img, grids, align_corners=False, mode='bilinear', padding_mode='zeros')


def blur_synthesis(model_output, sharp_img, control_params=None, compensation_net=None):
    B, C, H, W = sharp_img.shape
    num_poses = model_output.shape[1]
    
    if control_params:
        disps = scale_rotate(model_output, control_params)
    else:
        disps = model_output
    
    # [B, C, H, W] -> [B * num_poses, C, H, W]
    sharp_img_expanded = sharp_img.unsqueeze(1).expand(B, num_poses, C, H, W)
    sharp_img_expanded = sharp_img_expanded.reshape(B * num_poses, C, H, W)
    disps_reshaped = disps.reshape(B * num_poses, H, W, 2)
    
    warped = grid_sample(
        sharp_img_expanded, disps_reshaped
    ).reshape(B, num_poses, C, H, W)
    
    warped = torch.clamp(warped, -1, 1)
    warped_reshaped = warped.reshape(B * num_poses, C, H, W)
    
    cycle_warped = grid_sample(
        warped_reshaped, -disps_reshaped
    ).reshape(B, num_poses, C, H, W)
    
    warped_mean = warped.mean(dim=1)
    
    results = {
        'pred_blur': warped_mean,
        'cycle_warped_img': cycle_warped,
        'disps': disps
    }
    
    if compensation_net is not None:
        results['pred_blur_comp'] = compensation_net(warped_mean)
    
    return results


@torch.no_grad
def blur_data_augmentation(blur_img, sharp_img, model):
    B = blur_img.size()[0]
    control_params = (
        torch.empty(B).uniform_(1.0, 2.0),
        torch.empty(B).uniform_(-1.0, 1.0) * (math.pi / 2.0)
    )

    target_shuf = sharp_img[torch.randperm(B)]
    results = blur_synthesis(model(blur_img), blur_img, target_shuf, control_params)
    
    return results['pred_blur']
