#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J BERT_8BIT
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/BERT/

echo 'BERT-LARGE (Baseline - AdamW 32-bit)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base_32bit \
    --model bert_large \
    --max_length 128 \
    --batch_size 32

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base_32bit \
    --model bert_large \
    --max_length 128

sleep 10

echo 'BERT-LARGE (AdamW 8-bit)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --optim_8bit

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 128
CONTAINER

cleanup_slurm_logs