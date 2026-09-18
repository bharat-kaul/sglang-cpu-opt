#!/bin/bash
# Validate the CORRECT tp>1 headroom approach on the fast 4-layer proxy:
#   SGLANG_CPU_OMP_THREADS_BIND leaves ~2 cores/rank free at LAUNCH (fixed thread count -> shm
#   barriers sized right, NO per-forward set_num_threads -> NO deadlock), avoiding the M=1 decode
#   thread cliff without the tp>1-unsafe per-forward cap. + gloo for the all-reduce.
# Dumps `numactl --hardware` first so we can confirm/correct the per-domain core ranges.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --job-name=proxy_tp4_bind
#SBATCH --output=/scratch/bkaul/proxy_tp4_bind_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
echo "=== NUMA LAYOUT ==="; numactl --hardware | grep -E 'available|cpus'
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_TP_GLOO=1                    # gloo all-reduce (all-reduce latency lever)
# NO INTEL_CPU_DSV4_DECODE_THREADS: per-forward cap deadlocks at tp>1. Leave headroom at launch:
# one |-group per rank, each = its NUMA domain's cores MINUS ~2 (below the M=1 decode thread cliff).
export SGLANG_CPU_OMP_THREADS_BIND="0-39|42-82|85-125|128-168"
export INTEL_CPU_DSV4_TIMEIT=1
PROXY=/scratch/bkaul/models/flash-v4.1-shallow4
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$PROXY" --load-format dummy \
  --device cpu --tp 4 --batch-size 1 --input-len 1 --output-len 16 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.30 --max-total-tokens 8192
echo "[exit=$?]"
