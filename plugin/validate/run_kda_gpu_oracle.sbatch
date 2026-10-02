#!/bin/bash
# KDA GPU-oracle parity (1 GPU, podman + official sglang CUDA image): CPU reference
# recurrence (kda_linear_attention_cpu, pure torch) vs the real Triton
# fused_recurrent_kda. Independent-spec validation of the sole authoring gap.
# The CPU venv is torch+cpu (no CUDA), so the GPU kernel must run in the container.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=glm_kda_oracle
#SBATCH --output=/scratch/bkaul/glm_kda_oracle_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
nvidia-smi -L
DEVARGS=""; for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; done
# PYTHONPATH: plugin first (CPU reference import); container ships its own sglang for the
# Triton kernel. If the image's fused_recurrent_kda API has drifted, prepend the local
# sglang python path here instead.
podman run --rm $DEVARGS \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/test_kda_gpu_oracle.py 2>&1 | tail -40
echo "[exit=$?]"
