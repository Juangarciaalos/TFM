#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J GPT2_COMPARATION
#SBATCH --cpus-per-task=2
#SBATCH --output=salidas/slurm_%x_%j.out
#SBATCH --error=salidas/slurm_%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

RUN_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
HOST_OUTPUT_DIR="$WORK/salidas/$RUN_NAME"

mkdir -p "$HOST_OUTPUT_DIR"

exec > >(tee "$HOST_OUTPUT_DIR/${RUN_NAME}.out") 2> >(tee "$HOST_OUTPUT_DIR/${RUN_NAME}.err" >&2)

singularity exec --nv -B "$WORK":/workspace "$IMG" bash -lc "
    set -euo pipefail

    source /workspace/.venv/bin/activate
    
    export TFM_OUTPUT_DIR=\"/workspace/salidas/$RUN_NAME\"

    cd /workspace/modelos/GPT-2/

    echo 'GPT2-BASE (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_base --max_length 512 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_base --max_length 512

    sleep 10 

    echo 'GPT2-MEDIUM (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 512 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_medium --max_length 512

    sleep 10

    echo 'GPT2-MEDIUM OPTIMIZADO (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_medium --max_length 512 --batch_size 8 --amp --checkpointing
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_medium --max_length 512

    sleep 10

    echo 'GPT2-LARGE (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name base --model gpt2_large --max_length 512 --batch_size 8
    python plot_performance.py --job_id \$SLURM_JOB_ID --name base --model gpt2_large --max_length 512
    
    sleep 10

    echo 'GPT2-LARGE OPTIMIZADO (Secuencia 512)'
    python train.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_large --max_length 512 --batch_size 8 --amp --checkpointing
    python plot_performance.py --job_id \$SLURM_JOB_ID --name opt --model gpt2_large --max_length 512
"

rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true
rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" 2>/dev/null || true