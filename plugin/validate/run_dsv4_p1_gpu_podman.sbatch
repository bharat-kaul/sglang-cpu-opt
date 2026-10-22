#!/bin/bash
# P1 GPU REFERENCE leg (podman + official sglang CUDA image): run paired_eval.py on the FROZEN token-ID
# set via NATIVE CUDA (no plugin) = the trusted reference for the GPU/CPU-eager/CPU-C equivalence. The
# CPU venv has no CUDA, so the GPU leg must run in the CUDA container.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_p1_gpu
#SBATCH --output=/scratch/bkaul/dsv4_p1_gpu_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== exposing $N GPUs, tp=$N ==="
# NATIVE: no SGLANG_EXTERNAL_MODEL_PACKAGE, no INTEL_CPU_* -> pure CUDA reference. paired_eval.py adds
# its own dir to sys.path so `from task_gsm8k import ...` resolves; frozen file + model under /scratch.
podman run --rm $DEVARGS --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e MAX_TOTAL_TOKENS=8192 -e SWA_RATIO=0.8 \
  "$IMG" \
  python $WT/plugin/validate/paired_eval.py \
    --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
    --device cuda --tp $N --leg gpu --max-new 512 --mem-fraction 0.85 \
    --out-prefix $WT/plugin/validate/results/p1_paired 2>&1 | tail -40
echo "[exit=${PIPESTATUS[0]}] p1 gpu reference leg"
