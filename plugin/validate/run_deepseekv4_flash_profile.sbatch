#!/bin/bash
# Thesis-2 kernel-authoring GATE: authoritative per-op baseline on REAL DeepSeek-V4-Flash.
# The tiny profile is not reliable for the AMX-vs-torch-DSA split (tiny weights fit in cache,
# launch overhead dominates), so sample the real model at scale. py-spy --subprocesses follows
# the tp=4 worker processes that cProfile (main-process only) cannot see. Output-len 2 keeps it
# to prefill + 2 decode per round (still ~1 decode/token dominates wall time).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_flash_prof
#SBATCH --output=/scratch/bkaul/dsv4_flash_prof_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
OUT=/scratch/bkaul/flash_pyspy_${SLURM_JOB_ID}
cd /data/nfs_home/bkaul
# py-spy launches + samples the whole process tree (main + tp workers).
py-spy record --subprocesses --rate 25 --format speedscope --output "${OUT}.speedscope.json" -- \
  python -m sglang.bench_one_batch \
    --model-path "$CKPT" \
    --device cpu --tp 4 --batch-size 1 --input-len 8 --output-len 2 \
    --trust-remote-code --disable-cuda-graph \
    --mem-fraction-static 0.85 --max-total-tokens 16384 --swa-full-tokens-ratio 0.5
echo "[pyspy exit=$?] wrote ${OUT}.speedscope.json"
echo "[exit=$?]"
