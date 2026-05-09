#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J FINAL_GPT2
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/GPT-2/

echo 'G1 - GPT2-BASE L512 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_base \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_base \
    --max_length 512

sleep 10

echo 'G2 - GPT2-MEDIUM L512 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_medium \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_medium \
    --max_length 512

sleep 10

echo 'G3 - GPT2-MEDIUM L512 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_medium \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_medium \
    --max_length 512

sleep 10

echo 'G4 - GPT2-LARGE L512 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'G5 - GPT2-LARGE L512 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'G6 - GPT2-LARGE L512 SDPA'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100 \
    --sdpa

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa \
    --model gpt2_large \
    --max_length 512

sleep 10

echo 'G7 - GPT2-LARGE L512 SDPA + AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name sdpa_opt \
    --model gpt2_large \
    --max_length 512 \
    --batch_size 8 \
    --max_batches 100 \
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