#!/bin/bash
# COMBINED GPU job (one h200 allocation = one queue wait): both GPU-dependent tasks.
#   RUN A (post-trained, Option 2): teacher-forced per-position logit capture -> fidelity reference.
#   RUN B (Base/FP8, Option 1): gsm8k 8-shot EM -> reproduce published 90.8 (validate harness+infra).
# Two model loads (different checkpoints) but ONE node grab. Native CUDA sglang in the container (no
# plugin / no CPU env).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=dsv4_gpu_combined
#SBATCH --output=/scratch/bkaul/dsv4_gpu_combined_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
PROOT=/scratch/bkaul/podman/root; PRUN=/scratch/bkaul/podman/runroot; mkdir -p "$PROOT" "$PRUN"
PG="--root $PROOT --runroot $PRUN --events-backend=file --cgroup-manager=cgroupfs"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
RDIR=$WT/plugin/validate/results
nvidia-smi -L
DEV=""; N=0
for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do DEV="$DEV --device nvidia.com/gpu=$u"; N=$((N+1)); done
echo "=== exposing $N GPUs, tp=$N ==="
PODMAN="podman $PG run --rm $DEV --ipc=host --network=host -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0"

# ---- RUN A: Option 2 fidelity capture on POST-TRAINED ----
# Settings ALIGNED to the CPU production capture (SWA_RATIO=0.5, CONTEXT_LEN=4096) so the only
# difference vs CPU is the BACKEND — a SWA/context mismatch would change the attention mask and
# manufacture a false divergence.
echo "########## RUN A: Option 2 logit capture (post-trained) ##########"
$PODMAN -e CONTEXT_LEN=4096 -e MAX_TOTAL_TOKENS=131072 -e SWA_RATIO=0.5 "$IMG" \
  python $WT/plugin/validate/capture_logits.py \
    --frozen $RDIR/p1_frozen_tokenids.json \
    --device cuda --tp $N --leg gpu --topk 5 --max-seqs 6 --mem-fraction 0.85 \
    --out $RDIR/o2_logits_gpu.json 2>&1 | tail -20
echo "[RUN A exit=${PIPESTATUS[0]}]"

# ---- RUN B: Option 1 reproduce 90.8 on BASE (FP8) ----
echo "########## RUN B: Option 1 gsm8k 8-shot on Flash-Base (reproduce 90.8) ##########"
$PODMAN "$IMG" \
  python $WT/plugin/validate/task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash-Base \
    --data /scratch/bkaul/gsm8k_test.jsonl --device cuda --tp $N \
    --num-questions 1319 --num-shots 8 --chunk-size 64 --max-new-tokens 512 \
    --mem-fraction 0.85 --acc-min 0.30 --gens-schema-v2 \
    --out $RDIR/gpu_base_gsm8k.json \
    --gens-out /scratch/bkaul/gpu_base_gens.json 2>&1 \
  | grep -iE "chunk |acc=|accuracy|verdict|Error|Traceback|OOM"
echo "[RUN B exit=${PIPESTATUS[0]}]"
echo "[done] combined gpu job"
