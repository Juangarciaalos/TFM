#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J RESNET_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/ResNet/

echo 'RESNET-50 BASELINE (224x224)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet50 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet50 \
    --image_size 224

sleep 10

echo 'RESNET-50 OPTIMIZADO (224x224)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet50 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet50 \
    --image_size 224

sleep 10

echo 'RESNET-101 BASELINE (224x224)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model resnet101 \
    --image_size 224

sleep 10

echo 'RESNET-101 OPTIMIZADO (224x224)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 224 \
    --epochs 1 \
    --batch_size 64 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model resnet101 \
    --image_size 224

sleep 10

echo 'RESNET-101 BASELINE ALTA RESOLUCION (512x512)'
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

echo 'RESNET-101 OPTIMIZADO ALTA RESOLUCION (512x512)'
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

echo 'RESNET-101 BASELINE (512x512) + GESTIÓN DE FRAGMENTACIÓN'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test \
    --model resnet101 \
    --image_size 512

sleep 10

echo 'RESNET-101 OPTIMIZADO (512x512) + GESTIÓN DE FRAGMENTACIÓN'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test_opt \
    --model resnet101 \
    --image_size 512 \
    --epochs 1 \
    --batch_size 64 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_test_opt \
    --model resnet101 \
    --image_size 512

unset PYTORCH_CUDA_ALLOC_CONF
CONTAINER

cleanup_slurm_logs