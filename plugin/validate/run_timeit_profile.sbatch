#!/bin/bash
# Hotspot profile: per-op wall-time breakdown on real DeepSeek-V4-Flash (native MXFP4 MoE, tp=1),
# via the plugin's env-gated timers (INTEL_CPU_DSV4_TIMEIT=1 -> [DSV4 TIMEIT] dump at exit).
# SHORT workload (a prefill + a few decode steps) => load-dominated (~10 min), NOT the ~1h gen run.
# NOTE: perf is NOT ISA-portable. EMR gives a DIRECTIONAL op-share read; rerun on GNR for roofline %.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_timeit
#SBATCH --output=/scratch/bkaul/dsv4_timeit_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_TIMEIT=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size 1 --input-len 32 --output-len 4 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 8192
echo "[exit=$?]"
echo "=== per-op breakdown (prefill + decode accumulated) ==="
grep "DSV4 TIMEIT" /scratch/bkaul/dsv4_timeit_${SLURM_JOB_ID}.log || true
