#!/bin/bash
# GLM-5.3 Flash gsm8k on the GPU REFERENCE (native sglang CUDA, real fp8 weights, tp=4, podman) —
# the EQUIVALENCE oracle for the CPU result: SAME harness + SAME protocol (8-shot, 256 max-new,
# greedy, same stops/extraction, full test set), only device=cuda. CPU≡GPU accuracy (within the
# documented W8A16-vs-W8A8 fp8-activation noise) establishes the CPU build is correct end-to-end.
# NO plugin env (SGLANG_EXTERNAL_MODEL_PACKAGE unset) => pure native-GPU path.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --job-name=glm5_gsm8k_gpu
#SBATCH --output=/scratch/bkaul/glm5_gsm8k_gpu_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== GLM GPU gsm8k oracle: tp=$N real weights, SAME protocol as CPU ==="
LROOT=/scratch/bkaul/podman-glm-store; LRUN=/tmp/podman-glm-rr-gsm8k; mkdir -p "$LROOT" "$LRUN"
PODMAN="podman --root $LROOT --runroot $LRUN --cgroup-manager=cgroupfs --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs --storage-opt overlay.ignore_chown_errors=true"
trap 'rm -rf "$LRUN"/* 2>/dev/null' EXIT
$PODMAN image exists "$IMG" || $PODMAN pull "$IMG"
# SAME gsm8k params as the CPU run (run_glm5_gsm8k.sbatch): 8-shot, 256 max-new, chunk 32, full set.
$PODMAN run --rm $DEVARGS --userns=host \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e MAMBA_RADIX_STRATEGY=no_buffer -e DISABLE_RADIX=1 -e CONTEXT_LEN=${CONTEXT_LEN:-8192} \
  -e MAX_RUNNING=${MAX_RUNNING:-64} \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/GLM-5.3-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --device cuda --tp $N \
    --num-questions ${NQ:-1319} --chunk-size ${CHUNK:-32} \
    --num-shots ${SHOTS:-8} --max-new-tokens ${MAXNEW:-256} \
    --mem-fraction ${MEM_FRAC:-0.9} --chunked-prefill-size ${CPS:-512} \
    --watchdog-timeout 7000 \
    --out /scratch/bkaul/glm5_gsm8k_gpu.json \
    --gens-out /scratch/bkaul/glm5_gsm8k_gpu_gens.json 2>&1 | tail -60
echo "=== GPU gsm8k result ==="; cat /scratch/bkaul/glm5_gsm8k_gpu.json 2>/dev/null
echo "[exit=$?]"
