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
# Validate the model source against the pinned hash BEFORE execution (R2-F4, fail-closed).
KSRC=/scratch/bkaul/models/DeepSeek-V4-Flash/inference/kernel.py
EXPECT_SHA=59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2
GOT_SHA=$(sha256sum "$KSRC" | cut -d' ' -f1)
[ "$GOT_SHA" = "$EXPECT_SHA" ] || { echo "FATAL: kernel.py sha $GOT_SHA != expected pin $EXPECT_SHA"; exit 1; }
# Allocation-aware device selection: honor CUDA_VISIBLE_DEVICES (SLURM gres), not ALL enumerated GPUs (R2-F4).
VIS="${CUDA_VISIBLE_DEVICES:-all}"
if [ "$VIS" = "all" ]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
else
  UUIDS=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v v=",$VIS," 'index(v, ","$1",")>0 {print $2}')
fi
DEVARGS=""; for u in $UUIDS; do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; done
[ -n "$DEVARGS" ] || { echo "FATAL: no allocated GPU resolved from CUDA_VISIBLE_DEVICES=$VIS"; exit 1; }
# Dedicated per-job store/runroot off the absent /run/user + cgroupfs + subuid-free overlay + host userns
# (the bare `podman run` hits '/run/user denied' on these nodes; this is the repo's working incantation).
LROOT=/scratch/bkaul/podman-stores/${SLURM_JOB_ID}; LRUN=/tmp/podman-runroot-${SLURM_JOB_ID}; mkdir -p "$LROOT" "$LRUN"
trap 'rm -rf "$LRUN"/* 2>/dev/null' EXIT
PODMAN="podman --root $LROOT --runroot $LRUN --cgroup-manager=cgroupfs --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs --storage-opt overlay.ignore_chown_errors=true"
# PYTHONPATH: the model snapshot (inference/kernel.py = the real TileLang sparse_attn primitive).
# Forward run/image identity into the container so the saved-io provenance is real, not a fallback (R2-F4).
$PODMAN run --rm $DEVARGS --userns=host \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e PYTHONPATH=/scratch/bkaul/models/DeepSeek-V4-Flash \
  -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" -e ORACLE_IMG="$IMG" -e ORACLE_KERNEL_SHA="$EXPECT_SHA" \
  "$IMG" \
  python /data/nfs_home/bkaul/dsv4-tree-pilot/plugin/validate/test_sparse_gpu_oracle.py 2>&1 | tail -40
rc=${PIPESTATUS[0]}   # podman's exit, not tail's (P1-F7)
echo "[exit=$rc]"
exit "$rc"
