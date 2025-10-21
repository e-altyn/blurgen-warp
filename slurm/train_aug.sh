#!/bin/bash

#SBATCH --job-name=blurgen
#SBATCH --output=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --error=/scratch/ayakovenko/users/29e_alt/blurgen-warp/.logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=batch

set -e

START_TIME=$(date +%s)
echo "Job started at $(date)"

srun \
  --container-image /scratch/ayakovenko/stuff/nvcr.io+nvidia+pytorch+24.08-py3.sqsh \
  --container-mounts=/scratch/ayakovenko/users/29e_alt:/workspace:rw \
  bash -c "cd /workspace/blurgen-warp; \
    pip install -r misc/requirements.txt; \
    torchrun --nproc_per_node=8 \
      main.py train_aug"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
printf -v ELAPSED_TIME "%02d:%02d:%02d" $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60))
echo "Job completed at $(date)"
echo "Total duration: $ELAPSED_TIME"