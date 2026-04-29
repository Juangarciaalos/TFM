#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J BERT_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/bert_comparation-%j.out
#SBATCH --error=salidas/bert_comparation-%j.err
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
    mkdir salidas/BERT_\$SLURM_JOB_ID

    echo 'BERT-BASE (Secuencia 128)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model bert_base --max_length 128 --batch_size 32
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model bert_base --max_length 128

    sleep 10

    echo 'BERT-LARGE (Secuencia 128)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model bert_large --max_length 128 --batch_size 32
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model bert_large --max_length 128

    sleep 10

    echo 'BERT-LARGE OPTIMIZADO(Secuencia 128)'
    python train.py --job_id \$SLURM_JOB_ID --name opt --model bert_large --max_length 128 --batch_size 32 --amp --checkpointing
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt --model bert_large --max_length 128

    sleep 10

    echo 'BERT-LARGE (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model bert_large --max_length 512 --batch_size 32
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model bert_large --max_length 512
    
    sleep 10

    echo 'BERT-LARGE OPTIMIZADO (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name opt --model bert_large --max_length 512 --batch_size 32 --amp --checkpointing
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt --model bert_large --max_length 512
"