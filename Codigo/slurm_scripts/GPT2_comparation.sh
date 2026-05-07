#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J GPT2_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/GPT-2/

echo 'GPT2-BASE (Secuencia 512)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_base \
    --max_length 512 \
    --batch_size 8

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_base \
    --max_length 512

sleep 10

echo 'GPT2-MEDIUM (Secuencia 512)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_medium \
    --max_length 512 \
    --batch_size 8

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model gpt2_medium \
    --max_length 512

sleep 10

echo 'GPT2-MEDIUM OPTIMIZADO (Secuencia 512)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_medium \
    --max_length 512 \
    --batch_size 8 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model gpt2_medium \
    --max_length 512

sleep 10

echo 'GPT2-LARGE (Secuencia 512)'
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

echo 'GPT2-LARGE OPTIMIZADO (Secuencia 512)'
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
CONTAINER

cleanup_slurm_logs