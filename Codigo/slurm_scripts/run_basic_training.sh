#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J RUN_BASIC_TRAINING
#SBATCH --cpus-per-task=2
#SBATCH --output=run_basic_training-%j.out
#SBATCH --error=run_basic_training-%j.err
#SBATCH --mail-type=NONE

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"
nvidia-smi || true

export PYTORCH_CUDA_ALLOC_CONF="backend:native,max_split_size_mb:128,garbage_collection_threshold:0.8"

singularity exec --nv \
  -B "$WORK":/workspace \
  "$IMG" \
  bash -lc "cd /workspace; python basic_training.py --epochs 1 --batch_size 128 --samples 10000"