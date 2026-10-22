#!/bin/bash
# Option 2 GPU reference leg (podman + sglang CUDA image): teacher-forced per-position logit capture on
# the SAME frozen token IDs, native CUDA (no plugin). The build-robust reference for CPU-vs-GPU top-1
# fidelity (teacher-forced top-1 only flips on a near-tie, so build/ULP diffs are tolerated; a systematic
# early divergence is a real difference).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_o2_gpu
#SBATCH --output=/scratch/bkaul/dsv4_o2_gpu_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
PROOT=/scratch/bkaul/podman/root; PRUN=/scratch/bkaul/podman/runroot; mkdir -p "$PROOT" "$PRUN"
PODMAN_GLOBAL="--root $PROOT --runroot $PRUN --events-backend=file --cgroup-manager=cgroupfs"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== exposing $N GPUs, tp=$N ==="
podman $PODMAN_GLOBAL run --rm $DEVARGS --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e MAX_TOTAL_TOKENS=8192 -e SWA_RATIO=0.8 \
  "$IMG" \
  python $WT/plugin/validate/capture_logits.py \
    --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
    --device cuda --tp $N --leg gpu --topk 5 --max-seqs 6 --mem-fraction 0.85 \
    --out $WT/plugin/validate/results/o2_logits_gpu.json 2>&1 | tail -30
echo "[exit=${PIPESTATUS[0]}] o2 gpu capture"
