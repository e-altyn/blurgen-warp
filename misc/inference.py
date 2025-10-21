import argparse
import os
import random
import numpy as np
import torch
from PIL import Image

# Import your modules
from models.nafnet import NAFNetGrid
from utils.dataset import GoProLoader
from utils.synthesis import blur_synthesis
from utils.utils import load_ckpt, visualize_blur_trajectories, canonical_2d
from config import TrainAugConfig


def load_model(checkpoint_path, cfg, device='cuda', num_poses=16):
    """Load pretrained model from checkpoint."""
    # Initialize model with same config as training
    model = NAFNetGrid(**cfg.nafnet_grid_params)
    model = model.to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        epoch, metrics = load_ckpt(checkpoint_path, model, device=device)
        print(f"Loaded checkpoint from epoch {epoch}")
        if metrics:
            print(f"Metrics: {metrics}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    return model


def load_custom_image(image_path, normalize=True):
    """Load a custom image from path."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img, dtype=np.float32)

    if normalize:
        # Normalize to [-1, 1] range
        img_array = (img_array / 255.0) * 2 - 1.0

    # Convert to CHW format
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

    return img_tensor.unsqueeze(0)  # Add batch dimension


def denormalize_image(img_tensor, data_range=2.0):
    """Convert tensor from [-1, 1] to [0, 255] for saving."""
    img = img_tensor.squeeze(0).cpu().numpy()
    img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    return img


@torch.no_grad()
def inference(model, sharp_img, blur_img=None, device='cuda', num_poses=16):
    """
    Run inference on images.

    Args:
        model: Pretrained NAFNetGrid model
        sharp_img: Sharp image tensor [1, 3, H, W]
        blur_img: Optional blur image tensor [1, 3, H, W] (if None, uses sharp_img)
        device: Device to run on
        num_poses: Number of poses for blur synthesis

    Returns:
        results: Dictionary containing predicted blur and displacements
    """
    model.eval()

    sharp_img = sharp_img.to(device)
    if blur_img is None:
        blur_img = sharp_img
    else:
        blur_img = blur_img.to(device)

    # Run model to get motion parameters
    r, t, displacements = model(blur_img)

    # Synthesize blur using the predicted motion
    results = blur_synthesis(
        (r, t, displacements),
        blur_img,
        sharp_img,
        control_params=None,
        num_poses=num_poses,
        compensation_net=None
    )

    return results


def run_inference(args):
    parser = argparse.ArgumentParser(description='Blur Generation Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sharp', type=str, default=None,
                        help='Path to custom sharp image (optional)')
    parser.add_argument('--blur', type=str, default=None,
                        help='Path to custom blur image (optional, if not provided uses sharp)')
    parser.add_argument('--dataset', action='store_true',
                        help='Use random image from dataset instead of custom images')
    parser.add_argument('--dataset_mode', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset mode to use if --dataset is set')
    parser.add_argument('--output', type=str, default='assets/generated_blur.png',
                        help='Output path for predicted blur image')
    parser.add_argument('--trajectory_output', type=str, default='assets/trajectories.png',
                        help='Output path for visualized trajectories')
    parser.add_argument('--num_poses', type=int, default=16,
                        help='Number of poses for blur synthesis')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for dataset sampling')
    parser.add_argument('--trajectory_spacing', type=int, default=32,
                        help='Spacing between trajectory points in visualization')
    parser.add_argument('--trajectory_colormap', type=str, default='viridis',
                        help='Colormap for trajectory visualization (or None for grayscale)')

    args = parser.parse_args(args)

    # Set random seed
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, cfg=TrainAugConfig(),
                       device=args.device, num_poses=args.num_poses)

    # Load images
    if args.dataset:
        print(f"Loading random image from {args.dataset_mode} dataset...")
        dataset = GoProLoader(mode=args.dataset_mode, patch_size=256, portion=1.0)

        # Get random sample
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        sharp_img = torch.from_numpy(sample['sharp']).unsqueeze(0)
        blur_img = torch.from_numpy(sample['blur']).unsqueeze(0)

        print(f"Loaded sample {idx} from dataset")
        print(f"Image shape: {sharp_img.shape}")
    else:
        if args.sharp is None:
            raise ValueError("Must provide --sharp or use --dataset")

        print(f"Loading custom sharp image from {args.sharp}...")
        sharp_img = load_custom_image(args.sharp, normalize=True)

        if args.blur is not None:
            print(f"Loading custom blur image from {args.blur}...")
            blur_img = load_custom_image(args.blur, normalize=True)
        else:
            print("No blur image provided, using sharp image for motion estimation")
            blur_img = sharp_img

        print(f"Image shape: {sharp_img.shape}")

    # Run inference
    print("Running inference...")
    results = inference(model, sharp_img, blur_img, device=args.device, num_poses=args.num_poses)

    # Get predicted blur
    pred_blur = results['pred_blur']
    displacements = results['disps']

    # Denormalize and save predicted blur
    pred_blur_img = denormalize_image(pred_blur)
    Image.fromarray(pred_blur_img).save(args.output)
    print(f"Saved predicted blur to {args.output}")

    # Visualize and save trajectories
    print("Visualizing trajectories...")
    _, _, H, W, _ = displacements.shape
    grids_np = (displacements.cpu() + canonical_2d(H, W, device='cpu', denormalized=False))[0].numpy()

    # Use colormap if specified, otherwise None for grayscale
    colormap = args.trajectory_colormap if args.trajectory_colormap.lower() != 'none' else None

    visualize_blur_trajectories(
        grids_np, 
        output_path=args.trajectory_output,
        spacing=args.trajectory_spacing,
        colormap=colormap
    )
    print(f"Saved trajectory visualization to {args.trajectory_output}")

    print("Inference complete!")
