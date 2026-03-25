#!/bin/bash
#SBATCH -p tomasulo-gpu
#SBATCH --chdir=/nas/hdd-0/singularity_images/jgalos/tfm
#SBATCH -J TEST_LOAD_DATASETS
#SBATCH --cpus-per-task=2
#SBATCH --output=test_load_datasets-%j.out
#SBATCH --error=test_load_datasets-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

singularity exec --nv \
  -B "$WORK":/workspace \
  "$IMG" \
  bash -lc "
    source /workspace/.venv/bin/activate

    cd /workspace

    echo '[INFO] Test Tiny ImageNet'
    python -c \"from load_image_net import load_tiny_imagenet; train_loader, val_loader = load_tiny_imagenet(batch_size=64); print(f'Tiny ImageNet loaded with {len(train_loader)} batches')\"

    echo '[INFO] Test GLUE SST-2'
    python -c \"from load_glue_sst2 import load_glue_sst2; train_loader, val_loader = load_glue_sst2(batch_size=16); print(f'GLUE SST-2 loaded with {len(train_loader)} batches')\"

    echo '[INFO] Test WikiText-2'
    python -c \"from load_wikitext2 import load_wikitext2; tr, te = load_wikitext2(128); print('OK wikitext', len(tr), len(te))\"

    echo '[INFO] Test COCO'
    python -c \"from load_coco import load_coco; train_loader, val_loader = load_coco(batch_size=2); print(f'COCO loaded with {len(train_loader)} batches')\"
    "