#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J GPT2_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/gpt2_comparation-%j.out
#SBATCH --error=salidas/gpt2_comparation-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

singularity exec --nv -B "$WORK":/workspace "$IMG" bash -lc "
    source /workspace/.venv/bin/activate
    
    cd /workspace/modelos/GPT2/
    
    mkdir -p salidas/GPT2_\$SLURM_JOB_ID

    echo 'GPT2-BASE (Secuencia 256)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_base --max_length 256 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_base --max_length 256

    sleep 10 # Pausa térmica para la GPU

    echo 'GPT2-MEDIUM (Secuencia 256)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 256 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 256

    sleep 10

    echo 'GPT2-MEDIUM (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 512 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 512
    
    sleep 10

    echo 'GPT2-MEDIUM OPTIMIZADO (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_medium --max_length 512 --batch_size 8 --amp --checkpointing
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_medium --max_length 512
"