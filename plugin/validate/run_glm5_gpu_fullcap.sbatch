#!/bin/bash
# GLM-5.3 Flash GPU REFERENCE + full-tensor per-layer FINGERPRINT capture (native sglang CUDA,
# real weights, tp=4, podman). Freezes per-(pass,op,layer,rank) tensors (decoder-layer hidden,
# MHC hc_post, logits) to FULLCAP_FILE so the CPU build can be diffed against it for parity
# (accuracy-oracle Layer 1). Runs IN PARALLEL with the CPU bring-up (different partition).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
#SBATCH --job-name=glm5_gpu_fullcap
#SBATCH --output=/scratch/bkaul/glm5_gpu_fullcap_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
FCAP=/scratch/bkaul/glm5_gpu_fullcap_${SLURM_JOB_ID}.pt
nvidia-smi -L
DEVARGS=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEVARGS="$DEVARGS --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== GLM GPU reference: tp=$N real weights, capture -> $FCAP ==="
podman run --rm $DEVARGS \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e MODEL_PATH=/scratch/bkaul/models/GLM-5.3-Flash -e DEVICE=cuda -e TP=$N \
  -e SMOKE_PROMPTS="The capital of France is|2 + 2 =" -e MAX_NEW=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  -e PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin \
  -e SGLANG_EXTERNAL_MODEL_PACKAGE=gpu_ref_hook -e GLM_GPU_FULLCAP=1 -e FULLCAP_FILE=$FCAP \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py 2>&1 | tail -40
echo "=== fingerprint keys ==="; python3 -c "import torch;d=torch.load('$FCAP',map_location='cpu');print(len(d),'tensors');[print(k,tuple(v.shape)) for k,v in list(d.items())[:20]]" 2>/dev/null || echo "no fingerprint saved"
echo "[exit=$?]"
