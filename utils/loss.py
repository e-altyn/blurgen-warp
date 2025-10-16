import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.synthesis import canonical_2d, blur_synthesis

    
class LaplacianRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, grids):
        # grids: [B, num_poses, H, W, 2]
        grids = (torch.clamp(grids, -1.0, 1.0) + 1.0) / 2.0
        
        center = grids[..., 1:-1, 1:-1, :]
        up     = grids[...,  :-2, 1:-1, :]
        down   = grids[...,   2:, 1:-1, :]
        left   = grids[..., 1:-1,  :-2, :]
        right  = grids[..., 1:-1,   2:, :]
        
        laplacian = 4 * center - (up + down + left + right)
        
        return torch.mean(torch.sum(laplacian**2, dim=(2, 3, 4)))



class GeometricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, cycle_warped_img, sharp_img):
        b, num_poses, c, h, w = cycle_warped_img.shape
        
        cycle_warped_img = cycle_warped_img.reshape(b * num_poses, c, h, w)
        sharp_img_reshaped = sharp_img.repeat_interleave(num_poses, dim=0)
        
        return self.l1(cycle_warped_img, sharp_img_reshaped)


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, grids):
        B, _, H, W, _ = grids.shape
        
        canonical = canonical_2d(H, W, grids.device)[None, ...]
        displacements = grids - canonical
        
        relative_disps = displacements[:, 1:] - displacements[:, :-1]  # [B, num_poses-1, H, W, 2]
        relative_disps_flat = torch.flatten(relative_disps, start_dim=2)
        
        cos_sims = F.cosine_similarity(
            relative_disps_flat[:, :-1],
            relative_disps_flat[:, 1:],
            dim=2 
        ) # [B, num_flows-2]
        
        return 2.0 - cos_sims.mean() # keeping >= 0
    

class FlowAlignLoss(nn.Module):
    def __init__(self, flow_threshold=0.01, eps=1e-8):
        super().__init__()
        self.flow_threshold = flow_threshold
        self.eps = eps
    
    def forward(self, grids, flow):
        _, _, H, W, _ = grids.shape
        
        canonical = canonical_2d(H, W, grids.device)[None, ...]
        displacements = grids - canonical
        
        flow_directions = flow[:, :2, :, :].permute(0, 2, 3, 1)[:, None, ...]  # [B, 1, H, W, 2]
        flow_mags = flow[:, 2, :, :][:, None, :, :]  # [B, 1, H, W]
        
        # normalizing disps (cosine similarity loss) 
        # disp_norm = torch.sqrt((displacements ** 2).sum(dim=-1, keepdim=True) + self.eps)
        # displacements = displacements / (disp_norm + self.eps)  # [B, num_poses, H, W, 2]
        
        dot_products = (displacements * flow_directions).sum(dim=-1)  # [B, num_poses, H, W]
        dot_products = dot_products * (flow_mags >= self.flow_threshold)
        
        loss_per_pixel = F.relu(-dot_products) # ignore sharp angles
        
        return loss_per_pixel.sum()        


class CompositeLoss(nn.Module):
    """!!! HARDCODED !!!"""
    def __init__(self, weights=[1, 0.1, 1, 1]):
        super().__init__()
        self.weights = weights
        self.blur_fn = nn.L1Loss()
        self.lap_fn = LaplacianRegularizationLoss()
        self.geo_fn = GeometricConsistencyLoss()
        self.comp_fn = nn.L1Loss()
    
    def forward(self, model, comp_net, blur_img, sharp_img, num_poses):       
        results = blur_synthesis(model, blur_img, sharp_img, (1, 0), num_poses, comp_net)
        
        total_loss = (
            self.weights[0] * self.blur_fn(results['pred_blur'], blur_img) +
            self.weights[1] * self.lap_fn(results['grids']) +
            self.weights[2] * self.geo_fn(results['cycle_warped_img'], sharp_img) +
            self.weights[3] * self.comp_fn(results['pred_blur_comp'], blur_img)
        )
        
        return results, total_loss
