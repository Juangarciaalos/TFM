#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J BERT_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/BERT/

echo 'BERT-BASE (Secuencia 128)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_base \
    --max_length 128 \
    --batch_size 32

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_base \
    --max_length 128

sleep 10

echo 'BERT-LARGE (Secuencia 128)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 128 \
    --batch_size 32

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 128

sleep 10

echo 'BERT-LARGE OPTIMIZADO (Secuencia 128)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 128

sleep 10

echo 'BERT-LARGE (Secuencia 512)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 512 \
    --batch_size 32

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 512

sleep 10

echo 'BERT-LARGE OPTIMIZADO (Secuencia 512)'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 512 \
    --batch_size 32 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 512
CONTAINER

cleanup_slurm_logs