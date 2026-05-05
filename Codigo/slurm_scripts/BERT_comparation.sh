#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J BERT_COMPARATION
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

    cd /workspace/modelos/BERT/

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

rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true
rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" 2>/dev/null || true