import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.synthesis import blur_synthesis, disps2grids

    
class LaplacianRegularizationLoss(nn.Module):
    """Penalizes local discontinuity"""
    def __init__(self):
        super().__init__()
    
    def forward(self, disps):
        # would be better to convert to pixel space and find new coef
        grids = disps2grids(disps)
        
        center = grids[..., 1:-1, 1:-1, :]
        up     = grids[..., :-2,  1:-1, :]
        down   = grids[..., 2:,   1:-1, :]
        left   = grids[..., 1:-1, :-2,  :]
        right  = grids[..., 1:-1, 2:,   :]
        
        laplacian = 4 * center - (up + down + left + right)
        
        # sum for (B, num_poses)
        return torch.mean(laplacian**2, dim=(2,3,4)).sum()



class GeometricConsistencyLoss(nn.Module):
    """Penalizes divergency and non-uniformity along vector directions"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, cycle_warped_img, sharp_img):
        b, num_poses, c, h, w = cycle_warped_img.shape
        
        cycle_warped_img = cycle_warped_img.reshape(b * num_poses, c, h, w)
        sharp_img_reshaped = sharp_img.repeat_interleave(num_poses, dim=0)
        
        return self.l1(cycle_warped_img, sharp_img_reshaped)


class TemporalSmoothnessLoss(nn.Module):
    """Promotes temoral order"""
    def __init__(self):
        super().__init__()
    
    def forward(self, disps):
        B, _, H, W, _ = disps.shape
        
        relative_disps = disps[:, 1:] - disps[:, :-1]  # [B, num_poses-1, H, W, 2]
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
    
    def forward(self, disps, flow):
        flow_directions = flow[:, :2, :, :].permute(0, 2, 3, 1)[:, None, ...]  # [B, 1, H, W, 2]
        flow_mags = flow[:, 2:, :, :]  # [B, 1, H, W]
        
        # normalizing disps (cosine similarity loss) 
        # disp_norm = torch.sqrt((disps ** 2).sum(dim=-1, keepdim=True) + self.eps)
        # disps = disps / (disp_norm + self.eps)  # [B, num_poses, H, W, 2]
        
        dot_products = (disps * flow_directions).sum(dim=-1)  # [B, num_poses, H, W]
        dot_products = dot_products * (flow_mags >= self.flow_threshold)
        
        loss_per_pixel = F.relu(-dot_products) # ignore sharp angles
        
        return loss_per_pixel.sum()        


class CompositeLoss(nn.Module):
    """!!! HARDCODED !!!"""
    def __init__(self, weights=None):
        super().__init__()
        self.loss_blur = nn.L1Loss()
        self.loss_lap = LaplacianRegularizationLoss()
        self.loss_geo = GeometricConsistencyLoss()
        self.loss_comp = nn.L1Loss()
        self.weights = weights if weights else [1, 0.1, 1, 1]
    
    def forward(self, model, comp_net, blur_img, sharp_img, num_poses):    
        results = blur_synthesis(model(blur_img), blur_img, sharp_img, None, num_poses, comp_net)
        
        total_loss = (
            self.weights[0] * self.loss_blur(results['pred_blur'], blur_img) +
            self.weights[1] * self.loss_lap(results['disps']) +
            self.weights[2] * self.loss_geo(results['cycle_warped_img'], sharp_img) +
            self.weights[3] * self.loss_comp(results['pred_blur_comp'], blur_img)
        )
        
        return results, total_loss
