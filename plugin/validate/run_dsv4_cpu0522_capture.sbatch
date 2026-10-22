#!/bin/bash
# VERSION-MATCHED CALIBRATION: run upstream sglang 0.5.22 NATIVELY on CPU (device=cpu, no plugin) in the
# container, capturing the frozen-token logits. Compare vs GPU-0.5.22 (same version/code) -> a PURE
# hardware CPU-vs-GPU signal. If ~100% -> divergence enters via our plugin/0.5.20; if ~82% -> inherent.
# Runs on a GPU node only because podman+crun is set up there (GPUs unused; device=cpu).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_cpu0522cap
#SBATCH --output=/scratch/bkaul/dsv4_cpu0522cap_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
PROOT=/scratch/bkaul/podman/root; PRUN=/scratch/bkaul/podman/runroot; mkdir -p "$PROOT" "$PRUN"
IMG=docker.io/lmsysorg/sglang:latest
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
# NATIVE 0.5.22 on CPU: SGLANG_USE_CPU_ENGINE=1, no plugin, no INTEL_CPU_*. Aligned SWA/context.
podman --root $PROOT --runroot $PRUN run --rm --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e SGLANG_USE_CPU_ENGINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CONTEXT_LEN=4096 -e MAX_TOTAL_TOKENS=131072 -e SWA_RATIO=0.5 \
  "$IMG" \
  python $WT/plugin/validate/capture_logits.py \
    --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
    --device cpu --tp 1 --leg cpu0522 --topk 5 --max-seqs 6 --mem-fraction 0.60 \
    --out $WT/plugin/validate/results/o2_logits_cpu0522.json 2>&1 | grep -iE "capture|sglang 0|Error|Traceback|not support|assert|NATIVE" | tail -25
echo "[exit=${PIPESTATUS[0]}]"
