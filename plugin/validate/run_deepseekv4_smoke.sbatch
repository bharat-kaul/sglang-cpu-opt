#!/bin/bash
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_run
#SBATCH --output=/scratch/bkaul/dsv4_run_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_BYPASS_DSA=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
cd /data/nfs_home/bkaul
OMP_NUM_THREADS=60 numactl --cpunodebind=0,1,2 --membind=0,1,2 python -m sglang.bench_one_batch \
  --model-path /scratch/$USER/models/flash-v4.1-tiny --load-format dummy \
  --device cpu --tp 1 --batch-size 1 --input-len 8 --output-len 4 \
  --trust-remote-code --disable-cuda-graph --mem-fraction-static 0.02 --max-total-tokens 8192
echo "[exit=$?]"
