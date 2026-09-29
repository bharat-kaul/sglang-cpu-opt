#!/bin/bash
# GPU REFERENCE via podman + official sglang image (native CUDA path, no plugin) = ground-truth oracle.
# Coherent output => CPU bug is integration-level (checkpoint fine); garbage => model/checkpoint suspect.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_gpu_ref
#SBATCH --output=/scratch/bkaul/dsv4_gpu_ref_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
IMG=docker.io/lmsysorg/sglang:latest
echo "=== GPUs visible to the job (cgroup) ==="; nvidia-smi -L
podman run --rm --device nvidia.com/gpu=all \
  -e CUDA_VISIBLE_DEVICES \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e ACC_DEVICE=cuda -e ACC_TP=4 -e ACC_QUICK=0 -e ACC_MAX_NEW=10 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py 2>&1 | tail -60
echo "[exit=$?]"
