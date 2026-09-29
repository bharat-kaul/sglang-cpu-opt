#!/bin/bash
# ISOLATION TEST: force the stock TORCH hc_pre path (hc_pre_torch_impl mixes) on the GPU,
# keeping the correct GPU sinkhorn/combine/post kernels. If coherent GPU output turns to
# GARBAGE => hc_pre_torch_impl (the mixes computation the CPU plugin relies on) is the bug.
# If it stays coherent => the mixes are fine and the CPU divergence is in sinkhorn/combine/post routing.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_gpu_torchpost
#SBATCH --output=/scratch/bkaul/dsv4_gpu_torchpost_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
echo "=== GPUs visible to the job (cgroup) ==="; nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== exposing $N GPUs by UUID, tp=$N ==="
podman run --rm $DEVARGS \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e ACC_DEVICE=cuda -e ACC_TP=$N -e ACC_QUICK=0 -e ACC_MAX_NEW=10 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e SGLANG_OPT_USE_TILELANG_MHC_PRE=0 \
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=0 \
  -e SGLANG_OPT_USE_FLASHINFER_MHC=0 \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py 2>&1 | tail -60
echo "[exit=$?]"
