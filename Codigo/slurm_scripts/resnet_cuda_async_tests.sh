#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J RESNET_CUDA_ASYNC
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/ResNet/

echo 'RESNET-101 512x512 baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'RESNET-101 512x512 AMP+CKPT'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'RESNET-101 512x512 cudaMallocAsync baseline'
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'RESNET-101 512x512 cudaMallocAsync AMP+CKPT'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async_opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async_opt \
    --model resnet101 \
    --image_size 512

unset PYTORCH_CUDA_ALLOC_CONF
CONTAINER

cleanup_slurm_logs