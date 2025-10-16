import argparse
import os

import torch
import torch.distributed as dist

from config import TrainAugConfig
from train_aug import launch_train_aug

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    args = parser.parse_args()

    if args.mode == "train_aug":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        launch_train_aug(cfg=TrainAugConfig())
        dist.destroy_process_group()
