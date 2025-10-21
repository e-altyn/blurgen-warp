import os
import time

import torch
import torch.distributed as dist

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    LearnedPerceptualImagePatchSimilarity as LPIPS,
    PeakSignalNoiseRatio as PSNR
)

from utils.utils import get_env, save_ckpt
from models.nafnet import NAFNetGrid, NAFNet
from utils.dataset import GoProLoader
from utils.loss import CompositeLoss


# ========================= SETUP FUNCTIONS ========================= #

def setup_training_components(model, comp_net, lr, weight_decay, total_iterations, loss_weights):
    """Initialize optimizer, scheduler, and loss function."""
    loss_fn = CompositeLoss(loss_weights)
    
    params = list(model.parameters()) + list(comp_net.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-7)  
    
    return loss_fn, optimizer, scheduler, params


def setup_metrics(cfg, device):
    """Initialize evaluation metrics."""
    ssim_metric = SSIM(data_range=cfg.data_range, sync_on_compute=True).to(device)
    lpips_metric = LPIPS(net_type="vgg", sync_on_compute=True).to(device)
    psnr_metric = PSNR(data_range=cfg.data_range, sync_on_compute=True).to(device)
    
    return ssim_metric, lpips_metric, psnr_metric


# ========================= TRAINING EPOCH ========================= #

def train_epoch(model, comp_net, train_loader, loss_fn, optimizer, 
                    scheduler, params, num_poses, device, epoch):
    model.train()
    comp_net.train()
    train_loader.sampler.set_epoch(epoch - 1)
    
    loss_sum = torch.tensor([0.0], device=device)
    
    for sample in train_loader:
        optimizer.zero_grad(set_to_none=True)
        
        blur_img = sample["blur"].to(device, non_blocking=True)
        sharp_img = sample["sharp"].to(device, non_blocking=True)
        
        _, loss = loss_fn(model, comp_net, blur_img, sharp_img, num_poses)
        loss_sum += loss.item()
        
        loss.backward()
        clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
    
    train_loss = sync_tensor(loss_sum / len(train_loader))
    return train_loss


# ========================= VALIDATION EPOCH ========================= #

def validate_epoch(model, comp_net, val_loader, loss_fn, metrics, num_poses, device):
    model.eval()
    comp_net.eval()
    
    ssim_metric, lpips_metric, psnr_metric = metrics
    val_loss_sum = torch.tensor([0.0], device=device)
    
    with torch.no_grad():
        for sample in val_loader:
            blur_img = sample["blur"].to(device, non_blocking=True)
            sharp_img = sample["sharp"].to(device, non_blocking=True)
            
            results, loss = loss_fn(model, comp_net, blur_img, sharp_img, num_poses)
            
            pred = torch.clamp(results['pred_blur'], -1, 1)
            del results
            
            psnr_metric.update(pred, blur_img)
            ssim_metric.update(pred, blur_img)
            lpips_metric.update(pred, blur_img)
            
            del pred, blur_img, sharp_img
            
            val_loss_sum += loss.item()
            del loss
    
    val_loss = sync_tensor(val_loss_sum / len(val_loader))
    psnr = sync_torchmetrics(psnr_metric)
    val_ssim = sync_torchmetrics(ssim_metric)
    val_lpips = sync_torchmetrics(lpips_metric)
    
    return val_loss, psnr, val_ssim, val_lpips


# ========================= HELPER FUNCTIONS ========================= #
    
def sync_tensor(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def sync_torchmetrics(metric):
    value = metric.compute().item()
    metric.reset()
    return value


def log_training(epoch, num_epochs, train_loss, elapsed_time, rank):
    if rank == 0:
        print(f"[{epoch}/{num_epochs}] Loss: {train_loss:.4f} | Time: {elapsed_time:.2f}")


def log_validation(epoch, num_epochs, val_loss, psnr, val_ssim, val_lpips, elapsed_time, rank):
    if rank == 0:
        print(
            f"[Val {epoch}/{num_epochs}] Val Loss: {val_loss:.4f} | PSNR: {psnr:.2f} | "
            f"LPIPS: {val_lpips:.4f} | SSIM: {val_ssim:.4f} | Time: {elapsed_time:.2f}"
        )


def save_best_val(save_dir, epoch, model, val_loss, val_lpips, val_ssim, 
                        best_val_loss, rank):
    if rank == 0 and val_loss < best_val_loss:
        save_ckpt(save_dir, epoch, model.module, loss=val_loss, 
                 lpips=val_lpips, ssim=val_ssim, name="best_val")
        return val_loss
    return best_val_loss


# ========================= MAIN TRAINING LOOP ========================= #

def train_aug(model, optimizer, scheduler, params, 
              loss_fn, comp_net, train_loader, val_loader, 
              num_poses, val_freq, save_dir, num_epochs, metrics):
    
    rank, world_size, local_rank, device = get_env()
    
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(
            model, comp_net, train_loader, loss_fn, optimizer, 
            scheduler, params, num_poses, device, epoch
        )
        log_training(epoch, num_epochs, train_loss, time.time() - start_time, rank)
        
        if epoch % val_freq == 0:
            val_start_time = time.time()
            val_loss, psnr, val_ssim, val_lpips = validate_epoch(
                model, comp_net, val_loader, loss_fn, metrics, num_poses, device
            )
            log_validation(
                epoch, num_epochs, val_loss, psnr, val_ssim, 
                val_lpips, time.time() - val_start_time, rank
            )
            
            best_val_loss = save_best_val(
                save_dir, epoch, model, val_loss, val_lpips, val_ssim, 
                best_val_loss, rank
            )
    
    return model, comp_net


# ========================= DATA LOADING ========================= #

def setup_dataloaders(cfg):
    train_dataset = GoProLoader("train", cfg.patch_size)
    val_dataset = GoProLoader("test", portion=cfg.val_portion)
    
    train_sampler = DistributedSampler(train_dataset, drop_last=True, seed=cfg.train_seed)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, 
        num_workers=cfg.num_workers, persistent_workers=True, 
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, val_loader


def setup_models(cfg, device, local_rank):
    """Initialize and prepare models for distributed training."""
    
    
    model = NAFNetGrid(**cfg.nafnet_grid_params).to(device)
    comp_net = NAFNet(**cfg.nafnet_comp_params).to(device)
    
    model = DDP(model, device_ids=[local_rank])
    comp_net = DDP(comp_net, device_ids=[local_rank])
    
    return model, comp_net


# ========================= LAUNCH ========================= #

def launch_train_aug(cfg):
    rank, world_size, local_rank, device = get_env()
    
    model, comp_net = setup_models(cfg, device, local_rank)
    train_loader, val_loader = setup_dataloaders(cfg)
    
    save_dir = os.path.join(cfg.checkpoint_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
    
    loss_fn, optimizer, scheduler, params = setup_training_components(
        model, comp_net, cfg.lr, cfg.weight_decay, cfg.num_epochs, None
    )
    metrics = setup_metrics(cfg, device)
    
    train_aug(
        model=model, optimizer=optimizer, scheduler=scheduler, params=params, loss_fn=loss_fn,
        comp_net=comp_net, train_loader=train_loader, val_loader=val_loader, num_poses=cfg.num_poses, 
        val_freq=cfg.val_freq, save_dir=save_dir, num_epochs=cfg.num_epochs, metrics=metrics
    )
