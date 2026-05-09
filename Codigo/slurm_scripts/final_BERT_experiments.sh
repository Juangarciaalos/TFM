#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J FINAL_BERT
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

source common/slurm_common.sh

run_in_container "/workspace/.venv" <<'CONTAINER'
cd /workspace/modelos/BERT/

echo 'B1 - BERT-BASE L128 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_base \
    --max_length 128 \
    --batch_size 32 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_base \
    --max_length 128

sleep 10

echo 'B2 - BERT-LARGE L128 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 128

sleep 10

echo 'B3 - BERT-LARGE L128 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 128

sleep 10

echo 'B4 - BERT-LARGE L512 Baseline'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 512 \
    --batch_size 32 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base \
    --model bert_large \
    --max_length 512

sleep 10

echo 'B5 - BERT-LARGE L512 AMP + Checkpointing'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 512 \
    --batch_size 32 \
    --max_batches 100 \
    --amp \
    --checkpointing

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt \
    --model bert_large \
    --max_length 512

sleep 10

echo 'B6 - BERT-LARGE L128 AdamW FP32'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name base_32bit \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --max_batches 100

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name base_32bit \
    --model bert_large \
    --max_length 128

sleep 10

echo 'B7 - BERT-LARGE L128 AdamW 8-bit'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 128 \
    --batch_size 32 \
    --max_batches 100 \
    --optim_8bit

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 128

sleep 10

echo 'B8 OPCIONAL - BERT-LARGE L512 AdamW 8-bit'
python train.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 512 \
    --batch_size 32 \
    --max_batches 100 \
    --optim_8bit

python plot_performance.py \
    --job_id "$SLURM_JOB_ID" \
    --name opt_8bit \
    --model bert_large \
    --max_length 512
CONTAINER

cleanup_slurm_logs