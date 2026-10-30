#!/bin/bash
# APPLES-TO-APPLES GPU TARGET: gsm8k 8-shot on the POST-TRAINED Flash checkpoint (same one the CPU
# plugin runs), native CUDA sglang in the container (no plugin). Protocol MATCHED to the CPU wave
# (num_shots=8, max_new_tokens=256, SWA_RATIO=0.5, CONTEXT_LEN=4096) so the ONLY difference vs the
# CPU 0.796 result is the BACKEND. Answers: is CPU 0.796 a real gap, or is GPU-Flash also ~0.80?
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_gpu_flash_acc
#SBATCH --output=/scratch/bkaul/dsv4_gpu_flash_acc_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
PROOT=/scratch/bkaul/podman/root; PRUN=/scratch/bkaul/podman/runroot; mkdir -p "$PROOT" "$PRUN"
PG="--root $PROOT --runroot $PRUN --events-backend=file --cgroup-manager=cgroupfs"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:v0.5.20-cu130   # version-matched to the CPU venv (0.5.20.dev373)
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
RDIR=$WT/plugin/validate/results
nvidia-smi -L
DEV=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEV="$DEV --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== exposing $N GPUs, tp=$N ==="
PODMAN="podman $PG run --rm $DEV --ipc=host --network=host -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0"

echo "########## GPU gsm8k 8-shot on POST-TRAINED Flash (apples-to-apples vs CPU 0.796) ##########"
$PODMAN -e CONTEXT_LEN=4096 -e MAX_TOTAL_TOKENS=131072 -e SWA_RATIO=0.5 "$IMG" \
  python $WT/plugin/validate/task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl --device cuda --tp $N \
    --num-questions 1319 --num-shots 8 --chunk-size 64 --max-new-tokens 256 \
    --mem-fraction 0.85 --acc-min 0.30 --gens-schema-v2 \
    --out $RDIR/gpu_flash_gsm8k.json \
    --gens-out /scratch/bkaul/gpu_flash_gens.json 2>&1 \
  | grep -iE "chunk |acc=|accuracy|verdict|Error|Traceback|OOM"
echo "[GPU-Flash gsm8k exit=${PIPESTATUS[0]}]"
echo "[done] gpu flash accuracy"
