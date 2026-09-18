#!/bin/bash
# Validate: does the per-forward decode thread cap work under tp>1 WITH gloo all-reduce?
# Uses the fast 4-layer perf proxy (dummy weights, loads in seconds) so we can confirm tp=4
# decode is fast (cap engaged, no shm-barrier deadlock) BEFORE spending an 8-min full-model load.
# Expected: "[DECODE THREADCAP] tp=4 with gloo ... ENABLED" and a small Decode median latency.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --job-name=proxy_tp4_gloo
#SBATCH --output=/scratch/bkaul/proxy_tp4_gloo_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_TP_GLOO=1                    # gloo all-reduce -> decode cap is tp>1-safe
export INTEL_CPU_DSV4_DECODE_THREADS=auto          # per-forward cap, now enabled at tp>1 w/ gloo
export INTEL_CPU_DSV4_TIMEIT=1
PROXY=/scratch/bkaul/models/flash-v4.1-shallow4
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$PROXY" --load-format dummy \
  --device cpu --tp 4 --batch-size 1 --input-len 1 --output-len 16 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.30 --max-total-tokens 8192
echo "[exit=$?]"
