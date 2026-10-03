#!/bin/bash
# GLM-5.3 Flash GPU DUMMY-weight per-layer FINGERPRINT (native sglang CUDA, podman, tp=1,
# TINY arch-faithful config + DETERMINISTIC dummy weights). Paired with the CPU dummy run on the
# SAME tiny config + SAME deterministic dummy init -> the two per-(pass,op,layer) captures diff
# apples-to-apples to prove the assembled CPU model's WIRING + kernel math matches the GPU
# reference, WITHOUT a 306GB real-weight load and WITHOUT exercising the fp8 bridge (that is the
# separate real-weight parity run vs the archived glm5_gpu_fullcap_REAL_381095.pt).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=glm5_gpu_dummy_fc
#SBATCH --output=/scratch/bkaul/glm5_gpu_dummy_fullcap_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
FCAP=/scratch/bkaul/glm5_gpu_dummy_fullcap_${SLURM_JOB_ID}.pt
nvidia-smi -L
# Single GPU (tiny config) — pick the first free device by UUID.
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
echo "=== GLM GPU DUMMY reference: tp=1 tiny+deterministic-dummy, capture -> $FCAP ==="
# PERSISTENT GLM-dedicated podman store on /scratch (pulled ONCE, reused): a per-job store re-pulls
# the ~36GB image every run (~7+ min fuse-overlayfs unpack onto NFS). GLM GPU jobs run serially, so a
# dedicated store avoids the shared-store stale-lock races AND the repeat pull. CLI --root does NOT
# inherit storage.conf's overlay opts, so pass the subuid-free unpack knobs explicitly; runroot is
# per-job on local /tmp (off the absent /run/user). --userns=host maps the container to the caller.
LROOT=/scratch/bkaul/podman-glm-store; LRUN=/tmp/podman-runroot-${SLURM_JOB_ID}; mkdir -p "$LROOT" "$LRUN"
PODMAN="podman --root $LROOT --runroot $LRUN --cgroup-manager=cgroupfs --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs --storage-opt overlay.ignore_chown_errors=true"
trap 'rm -rf "$LRUN" 2>/dev/null' EXIT  # keep the image store; only clean the per-job runroot
$PODMAN image exists "$IMG" || $PODMAN pull "$IMG"
$PODMAN run --rm --device nvidia.com/gpu=$U --userns=host \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e MODEL_PATH=/scratch/bkaul/models/glm5-tiny -e DEVICE=cuda -e TP=1 \
  -e LOAD_FORMAT=dummy -e DETERMINISTIC_DUMMY=1 \
  -e SMOKE_PROMPTS="The capital of France is" -e MAX_NEW=${MAX_NEW:-1} \
  -e DISABLE_RADIX=1 -e DISABLE_CUDA_GRAPH=1 \
  -e MEM_FRAC=${MEM_FRAC:-0.9} -e MAX_RUNNING=8 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin \
  -e SGLANG_EXTERNAL_MODEL_PACKAGE=gpu_ref_hook -e GLM_GPU_FULLCAP=1 -e FULLCAP_FILE=$FCAP \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py 2>&1 | tail -50
echo "=== dummy fingerprint (host has no torch; check file exists + size) ==="; ls -la "$FCAP" 2>/dev/null && echo "fingerprint SAVED ($(stat -c%s "$FCAP" 2>/dev/null) bytes) — diff offline with diff_fullcap.py" || echo "NO fingerprint file at $FCAP"
echo "[exit=$?]"
