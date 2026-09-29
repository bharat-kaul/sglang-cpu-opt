#!/bin/bash
# GPU REFERENCE + per-layer capture: native sglang GPU path with the GPU-safe hidden-state hook
# (SGLANG_EXTERNAL_MODEL_PACKAGE=gpu_ref_hook). Writes per-layer stats + logits to HID_DBG_FILE to
# diff against the CPU trace (same format) and localize the first divergent layer.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_gpu_captp
#SBATCH --output=/scratch/bkaul/dsv4_gpu_captp_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
HID=/scratch/bkaul/dsv4_gpu_hid_${SLURM_JOB_ID}.txt
nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
podman run --rm $DEVARGS \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e ACC_DEVICE=cuda -e ACC_TP=$N -e ACC_QUICK=1 -e ACC_MAX_NEW=1 -e ACC_PROMPT_IDX=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin \
  -e SGLANG_OPT_USE_TILELANG_MHC_PRE=0 -e SGLANG_OPT_USE_FLASHINFER_MHC=0 -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  -e SGLANG_EXTERNAL_MODEL_PACKAGE=gpu_ref_hook -e GPU_REF_HID_DEBUG=1 -e HID_DBG_FILE=$HID \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py 2>&1 | tail -30
echo "=== GPU PER-LAYER TRACE ==="; cat "$HID" 2>/dev/null || echo "no trace (hook did not install)"
echo "[exit=$?]"
