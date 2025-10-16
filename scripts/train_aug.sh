#!/bin/bash

#SBATCH --job-name=blurgen
#SBATCH --output=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --error=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=batch

set -e

echo "Job started at $(date)"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

srun \
  --container-image /scratch/ayakovenko/stuff/nvcr.io+nvidia+pytorch+24.08-py3.sqsh \
  --container-mounts=/scratch/ayakovenko/users/29e_alt:/workspace:rw \
  bash -c "cd /workspace/blurgen-warp; \
    pip install -r scripts/requirements.txt; \
    torchrun \
      --nproc_per_node=8 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
      main.py train_aug"

echo "Job completed at $(date)"
