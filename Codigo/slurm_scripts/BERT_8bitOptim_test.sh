#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J BERT_8BIT
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/bert_8bit-%j.out
#SBATCH --error=salidas/bert_8bit-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

singularity exec --nv -B "$WORK":/workspace "$IMG" bash -lc "
    source /workspace/.venv/bin/activate
    cd /workspace/modelos/BERT/

    mkdir -p salidas/BERT_\$SLURM_JOB_ID

    echo 'BERT-LARGE (Baseline - AdamW 32-bit)'
    python train.py --job_id \$SLURM_JOB_ID --name base_32bit --model bert_large --max_length 128 --batch_size 32
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base_32bit --model bert_large --max_length 128

    sleep 10

    echo 'BERT-LARGE (AdamW 8-bit)'
    python train.py --job_id \$SLURM_JOB_ID --name opt_8bit --model bert_large --max_length 128 --batch_size 32 --optim_8bit
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt_8bit --model bert_large --max_length 128
"