#!/bin/bash

#SBATCH --job-name=blurgen
#SBATCH --output=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --error=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32 # 8 GPUs * 4 workers per each
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --partition=batch

set -e

echo "Job started at $(date)"

srun \
  --container-image /scratch/ayakovenko/stuff/nvcr.io+nvidia+pytorch+24.08-py3.sqsh \
  --container-mounts=/scratch/ayakovenko/users/29e_alt:/workspace:rw \
  bash -c "cd /workspace/blurgen-warp; \
    pip install -r scripts/requirements.txt; \
    torchrun \
      --nnodes=1 \
      --nproc_per_node=8 \
      main.py train_aug"

echo "Job completed at $(date)"
