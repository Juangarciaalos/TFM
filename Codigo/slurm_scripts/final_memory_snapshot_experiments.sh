#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J FINAL_SNAPSHOT
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv_torch22" <<'CONTAINER'
echo 'S1 - Snapshot BERT-Large L512 AMP + Checkpointing'
cd /workspace/modelos/BERT/

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_snapshot \
    --model bert_large \
    --max_length 512 \
    --batch_size 4 \
    --amp \
    --checkpointing \
    --memory_snapshot \
    --snapshot_batches 3 \
    --snapshot_max_entries 20000

sleep 10

echo 'S2 - Snapshot GPT2-Large L512 AMP + Checkpointing'
cd /workspace/modelos/GPT-2/

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_snapshot \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 1 \
    --amp \
    --checkpointing \
    --memory_snapshot \
    --snapshot_batches 3 \
    --snapshot_max_entries 20000

sleep 10

echo 'S3 - Snapshot GPT2-Large L512 SDPA + AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa_opt_snapshot \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 1 \
    --sdpa \
    --amp \
    --checkpointing \
    --memory_snapshot \
    --snapshot_batches 3 \
    --snapshot_max_entries 20000

sleep 10

echo 'S4 - Snapshot ResNet101 512x512 AMP + Checkpointing'
cd /workspace/modelos/ResNet/

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_snapshot \
    --model resnet101 \
    --image_size 512 \
    --batch_size 32 \
    --amp \
    --checkpointing \
    --memory_snapshot \
    --snapshot_batches 3 \
    --snapshot_max_entries 20000

sleep 10

echo 'S5 - Snapshot ResNet101 512x512 max_split_size_mb=128 + AMP + Checkpointing'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name alloc_snapshot \
    --model resnet101 \
    --image_size 512 \
    --batch_size 32 \
    --amp \
    --checkpointing \
    --memory_snapshot \
    --snapshot_batches 3 \
    --snapshot_max_entries 20000

unset PYTORCH_CUDA_ALLOC_CONF
CONTAINER

cleanup_slurm_logs