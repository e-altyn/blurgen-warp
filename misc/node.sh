#!/bin/bash
srun -G 8 --cpus-per-task=32 --mem=128G --container-image \
  /scratch/ayakovenko/stuff/nvcr.io+nvidia+pytorch+24.08-py3.sqsh \
  --container-mounts=/scratch/ayakovenko/users/29e_alt:/workspace:rw --pty bash
