#!/bin/bash
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=q1175_gpu_trace
#SBATCH --output=/scratch/bkaul/q1175_gpu_trace_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
SOURCE="$ROOT/source/plugin"
RUNTIME="$ROOT/gpu_runtime_$SLURM_JOB_ID"
mkdir "$RUNTIME" "$RUNTIME/cdi" "$RUNTIME/run" "$RUNTIME/tmp"
export XDG_RUNTIME_DIR="$RUNTIME/run" TMPDIR="$RUNTIME/tmp"
chmod 700 "$XDG_RUNTIME_DIR"
nvidia-ctk cdi generate --format=yaml --output="$RUNTIME/cdi/nvidia.yaml"
IFS=',' read -r -a ALLOCATED <<< "${CUDA_VISIBLE_DEVICES:?No allocated GPUs}"
test "${#ALLOCATED[@]}" = 2
DEVICES=()
for device in "${ALLOCATED[@]}"; do
    uuid=$(nvidia-smi --id="$device" --query-gpu=uuid --format=csv,noheader)
    test -n "$uuid"
    DEVICES+=(--device "nvidia.com/gpu=$uuid")
done
podman --root /scratch/bkaul/podman/root --runroot /scratch/bkaul/podman/runroot \
    --cdi-spec-dir "$RUNTIME/cdi" --events-backend=file --cgroup-manager=cgroupfs \
    run --pull=never --rm "${DEVICES[@]}" --ipc=host -v /scratch:/scratch \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
    -e SLURM_JOB_ID="$SLURM_JOB_ID" -e PYTHONPATH="$SOURCE" \
    -e SGLANG_EXTERNAL_MODEL_PACKAGE=dsv4_prefix_trace \
    docker.io/lmsysorg/sglang:v0.5.20-cu130 \
    python -u "$SOURCE/validate/run_prefix_trace.py" "$ROOT/plan.json" "$ROOT/gpu_$SLURM_JOB_ID" --device cuda
test -s "$ROOT/gpu_$SLURM_JOB_ID/complete.json"
