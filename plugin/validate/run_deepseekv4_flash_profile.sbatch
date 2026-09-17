#!/bin/bash
# Thesis-2 kernel-authoring GATE: authoritative per-op baseline on REAL DeepSeek-V4-Flash.
# Uses the plugin's env-gated per-function wall timers (INTEL_CPU_DSV4_TIMEIT=1) instead of
# py-spy: deterministic, ~zero overhead, and tp=4-safe (each rank prints its own [DSV4 TIMEIT]
# breakdown at exit). py-spy could not sustain sampling across tp=4 x OMP (100+ threads).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_flash_timeit
#SBATCH --output=/scratch/bkaul/dsv4_flash_timeit_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_TIMEIT=1            # env-gated per-op wall timers (inert without this)
unset INTEL_CPU_DSV4_FP4_MOE_BF16
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 4 --batch-size 1 --input-len 8 --output-len 2 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 16384 --swa-full-tokens-ratio 0.5
echo "[exit=$?]  grep '[DSV4 TIMEIT]' for the per-op breakdown"
