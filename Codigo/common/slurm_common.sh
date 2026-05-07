module purge
module load singularity

BASE="/nas/hdd-0/singularity_images/jgalos"
IMG="$BASE/containers/pytorch.sif"
WORK="$BASE/tfm"

RUN_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
HOST_OUTPUT_DIR="$WORK/salidas/$RUN_NAME"

mkdir -p "$HOST_OUTPUT_DIR"

exec > >(tee "$HOST_OUTPUT_DIR/${RUN_NAME}.out") 2> >(tee "$HOST_OUTPUT_DIR/${RUN_NAME}.err" >&2)


cleanup_slurm_logs() {
    rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true
    rm -f "$WORK/salidas/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" 2>/dev/null || true
}

run_in_container() {
    local venv_path="${1:-/workspace/.venv}"

    local host_script="$HOST_OUTPUT_DIR/container_commands.sh"
    local container_script="/workspace/salidas/$RUN_NAME/container_commands.sh"

    cat > "$host_script"
    chmod +x "$host_script"

    singularity exec --nv -B "$WORK":/workspace "$IMG" bash -lc "
        set -euo pipefail

        source \"$venv_path/bin/activate\"

        export PYTHONPATH=\"/workspace:\${PYTHONPATH:-}\"
        export TFM_OUTPUT_DIR=\"/workspace/salidas/$RUN_NAME\"
        export SLURM_JOB_ID=\"$SLURM_JOB_ID\"
        export SLURM_JOB_NAME=\"$SLURM_JOB_NAME\"

        bash \"$container_script\"
    "
}