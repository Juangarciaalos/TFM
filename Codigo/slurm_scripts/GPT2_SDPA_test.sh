#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J GPT2_SDPA_TEST
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/GPT-2/

echo 'GPT2-LARGE BASELINE atención manual'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'GPT2-LARGE SDPA'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --sdpa

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'GPT2-LARGE AMP+CKPT atención manual'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'GPT2-LARGE SDPA + AMP+CKPT'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa_opt \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --sdpa \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa_opt \
    --model gpt2_large \
    --max_length 512
CONTAINER

cleanup_slurm_logs