#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J MEMORY_SNAPSHOT
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

    echo 'Snapshot BERT-Large L512 Optimizado'
    cd /workspace/modelos/BERT/
    python train.py --job_id \$SLURM_JOB_ID --name opt_snapshot --model bert_large --max_length 512 --batch_size 32 --amp --checkpointing --memory_snapshot --snapshot_batches 3

    echo 'Snapshot GPT2-Large L512 Optimizado'
    cd /workspace/modelos/GPT-2/
    python train.py --job_id \$SLURM_JOB_ID --name opt_snapshot --model gpt2_large --max_length 512 --batch_size 8 --amp --checkpointing --memory_snapshot --snapshot_batches 3

    echo 'Snapshot ResNet101 512 allocator test'
    cd /workspace/modelos/ResNet/
    export PYTORCH_ALLOC_CONF=max_split_size_mb:128
    python train.py --job_id \$SLURM_JOB_ID --name alloc_snapshot --model resnet101 --image_size 512 --batch_size 64 --amp --checkpointing --memory_snapshot --snapshot_batches 3
    unset PYTORCH_ALLOC_CONF
"

rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true
rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" 2>/dev/null || true