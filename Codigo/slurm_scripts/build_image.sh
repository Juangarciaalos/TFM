#!/bin/bash
#SBATCH -p mendel-q
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos
#SBATCH -J BUILD-SIF
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=NONE

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
mkdir -p "$BASE"/{containers,cache,tmp,logs}

export SINGULARITY_CACHEDIR="$BASE/cache"
export SINGULARITY_TMPDIR="$BASE/tmp"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

singularity pull --disable-cache \
  --dir "$BASE/containers" pytorch.sif \
  docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ls -lh "$BASE/containers/pytorch.sif"