#!/bin/bash
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_smoke
#SBATCH --output=/scratch/bkaul/dsv4_smoke_%j.log
# DeepSeek-V4 (DSA) make-it-work smoke on AMX: does the tiny-arch model run a full
# prefill+decode end-to-end on CPU with the intel_amx backend + fp8 KV? This is the
# GNR-only continuation of the login-node make-it-work ladder (login node lacked AMX,
# which fp8 KV cache + the dsv4 attention backend require).
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
cd /data/nfs_home/bkaul/sglang-cpu-opt

TINY=/scratch/$USER/models/flash-v4.1-tiny
# Single socket (SNC-3 nodes 0,1,2), tp 1 — tiny model, just exercising code paths.
BIND="numactl --cpunodebind=0,1,2 --membind=0,1,2"

echo "[amx check] $(date +%T)"
python -c "import torch; print('amx_tile_supported=', torch.cpu._is_amx_tile_supported() if hasattr(torch.cpu,'_is_amx_tile_supported') else 'n/a')"

echo "[dsv4 tiny make-it-work smoke] $(date +%T)"
OMP_NUM_THREADS=60 $BIND python -m sglang.bench_one_batch \
  --model-path "$TINY" --load-format dummy --device cpu --tp 1 \
  --batch-size 1 --input-len 8 --output-len 4 \
  --trust-remote-code --disable-cuda-graph

echo "[exit=$?] $(date +%T)"
