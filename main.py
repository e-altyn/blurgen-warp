import argparse
import os

import torch
import torch.distributed as dist

from config import TrainAugConfig
from train_aug import launch_train_aug
from misc.inference import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    args, remaining_args = parser.parse_known_args()

    if args.mode == "train_aug":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        launch_train_aug(cfg=TrainAugConfig())
        dist.destroy_process_group()
    elif args.mode == "inference":
        run_inference(remaining_args)