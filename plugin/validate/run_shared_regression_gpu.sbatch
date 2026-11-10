#!/bin/bash
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=shared_native_gpu
#SBATCH --output=/scratch/bkaul/shared_native_gpu_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root required}")
SOURCE=$(realpath "${2:-$ROOT/gpu_source/plugin}")
/scratch/bkaul/venvs/sglang-cpu/bin/python "$SOURCE/validate/prepare_shared_regression_gpu.py" "$ROOT"
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
    run --rm "${DEVICES[@]}" --ipc=host -v /scratch:/scratch \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
    -e SLURM_JOB_ID="$SLURM_JOB_ID" -e DSV4_SHARED_GPU_ROOT="$ROOT" \
    -e DSV4_SHARED_GPU_QUANT_CAPTURE="${DSV4_SHARED_GPU_QUANT_CAPTURE:-0}" \
    -e PYTHONPATH="$SOURCE" -e SGLANG_EXTERNAL_MODEL_PACKAGE=shared_regression_gpu \
    docker.io/lmsysorg/sglang:v0.5.20-cu130 \
    python -u "$SOURCE/validate/run_shared_regression_gpu.py" "$ROOT"
/scratch/bkaul/venvs/sglang-cpu/bin/python "$SOURCE/validate/compare_shared_regression_gpu.py" "$ROOT" "$SLURM_JOB_ID"
