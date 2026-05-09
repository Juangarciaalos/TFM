#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J FINAL_RESNET
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/ResNet/

echo 'R1 - ResNet50 224x224 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet50 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet50 \
    --image_size 224

sleep 10

echo 'R2 - ResNet50 224x224 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet50 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet50 \
    --image_size 224

sleep 10

echo 'R3 - ResNet101 224x224 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 224

sleep 10

echo 'R4 - ResNet101 224x224 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 224

sleep 10

echo 'R5 - ResNet101 512x512 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'R6 - ResNet101 512x512 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'R7 - ResNet101 512x512 max_split_size_mb=128'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'R8 - ResNet101 512x512 max_split_size_mb=128 + AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test_opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test_opt \
    --model resnet101 \
    --image_size 512

unset PYTORCH_CUDA_ALLOC_CONF

sleep 10

echo 'R9 - ResNet101 512x512 cudaMallocAsync'
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'R10 - ResNet101 512x512 cudaMallocAsync + AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name cuda_async_opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --max_batches 100 \
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