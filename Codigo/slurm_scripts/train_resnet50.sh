#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J TRAIN_RESNET50
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/train_resnet50-%j.out
#SBATCH --error=salidas/train_resnet50-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

singularity exec --nv \
  -B "$WORK":/workspace \
  "$IMG" \
  bash -lc "
    source /workspace/.venv/bin/activate

    cd /workspace/modelos/ResNet50

    python train.py \$SLURM_JOB_ID
    python plot_memory.py \$SLURM_JOB_ID
    "