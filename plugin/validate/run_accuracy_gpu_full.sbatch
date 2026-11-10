#!/bin/bash
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=dsv4_matched_gpu_acc
#SBATCH --output=/scratch/bkaul/dsv4_matched_gpu_acc_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
RUNTIME="$ROOT/gpu_runtime_$SLURM_JOB_ID"
mkdir "$RUNTIME" "$RUNTIME/cdi" "$RUNTIME/run" "$RUNTIME/tmp"
export XDG_RUNTIME_DIR="$RUNTIME/run" TMPDIR="$RUNTIME/tmp"
chmod 700 "$XDG_RUNTIME_DIR"
nvidia-ctk cdi generate --format=yaml --output="$RUNTIME/cdi/nvidia.yaml"
IFS=',' read -r -a ALLOCATED <<< "${CUDA_VISIBLE_DEVICES:?No allocated GPUs}"
test "${#ALLOCATED[@]}" = 4
DEVICES=()
for device in "${ALLOCATED[@]}"; do
    uuid=$(nvidia-smi --id="$device" --query-gpu=uuid --format=csv,noheader)
    test -n "$uuid"
    DEVICES+=(--device "nvidia.com/gpu=$uuid")
done
IMAGE=docker.io/lmsysorg/sglang:v0.5.20-cu130
podman --root /scratch/bkaul/podman/root --runroot /scratch/bkaul/podman/runroot \
    --events-backend=file --cgroup-manager=cgroupfs image inspect "$IMAGE" \
    --format '{{.Id}}' > "$ROOT/gpu/image_id.txt"
podman --root /scratch/bkaul/podman/root --runroot /scratch/bkaul/podman/runroot \
    --cdi-spec-dir "$RUNTIME/cdi" --events-backend=file --cgroup-manager=cgroupfs \
    run --pull=never --rm "${DEVICES[@]}" --ipc=host -v /scratch:/scratch \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
    -e SGLANG_USE_CPU_ENGINE=0 -e SGLANG_EXTERNAL_MODEL_PACKAGE= \
    -e PYTHONUNBUFFERED=1 -e SLURM_JOB_ID="$SLURM_JOB_ID" -e EVIDENCE_ROOT="$ROOT" \
    "$IMAGE" bash -c '
        set -euo pipefail
        for shard in {0..15}; do
            bash "$EVIDENCE_ROOT/source/plugin/validate/run_accuracy_shard.sh" \
                "$EVIDENCE_ROOT" cuda "$shard" 4
        done
    '
for shard in {0..15}; do test -f "$ROOT/gpu/shard_$shard/complete"; done
echo "GPU_FULL_ACCURACY_COMPLETE job=$SLURM_JOB_ID"
