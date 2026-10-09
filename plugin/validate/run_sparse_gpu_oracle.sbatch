#!/bin/bash
# Sparse-attention GPU-oracle parity (1 GPU, podman + official sglang CUDA image): the REAL TileLang
# sparse_attn (DeepSeek-V4-Flash inference/kernel.py) vs our FP32 + BF16 torch oracle replicas.
# Resolves the open conformance question (is published sparse_attn bf16?) and saves seeded io so the
# CPU side can diff our native-AMX sparse_attend against the real kernel. Pattern == run_kda_gpu_oracle.
#SBATCH --job-name=sparse_gpu_oracle
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/sparse_gpu_oracle_%j.log
set -x
set -o pipefail   # P1-F7: so a container failure through the `| tail` pipe is NOT masked as exit 0
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
nvidia-smi -L
DEVARGS=""; for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; done
# Dedicated per-job store/runroot off the absent /run/user + cgroupfs + subuid-free overlay + host userns
# (the bare `podman run` hits '/run/user denied' on these nodes; this is the repo's working incantation).
LROOT=/scratch/bkaul/podman-stores/${SLURM_JOB_ID}; LRUN=/tmp/podman-runroot-${SLURM_JOB_ID}; mkdir -p "$LROOT" "$LRUN"
trap 'rm -rf "$LRUN"/* 2>/dev/null' EXIT
PODMAN="podman --root $LROOT --runroot $LRUN --cgroup-manager=cgroupfs --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs --storage-opt overlay.ignore_chown_errors=true"
# PYTHONPATH: the model snapshot (inference/kernel.py = the real TileLang sparse_attn primitive).
$PODMAN run --rm $DEVARGS --userns=host \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e PYTHONPATH=/scratch/bkaul/models/DeepSeek-V4-Flash \
  "$IMG" \
  python /data/nfs_home/bkaul/dsv4-tree-pilot/plugin/validate/test_sparse_gpu_oracle.py 2>&1 | tail -40
rc=${PIPESTATUS[0]}   # podman's exit, not tail's (P1-F7)
echo "[exit=$rc]"
exit "$rc"
