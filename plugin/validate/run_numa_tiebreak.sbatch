#!/bin/bash
# Tiebreaker: is the 3x decode gap between harnesses the NUMA binding? Run the SAME harness
# (throughput_bench, varied tokens) at 4L B=32 S=128 TWICE: single-NUMA (numactl -N0 -m0, the
# canonical per-rank bind) vs dual-NUMA (no bind, both sockets). Decode is BW-bound, so if NUMA is
# the cause single-NUMA ~= 30 tok/s (matches bench_wall_time) and dual ~= 90 (matches the ledger).
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_numatie
#SBATCH --output=/scratch/bkaul/dsv4_numatie_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=32
cd /data/nfs_home/bkaul
COMMON="--model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json --tp 1 --batch 32 --input-len 128 --output-len 8 --layers 4 --load-format dummy --mem-fraction 0.85"
echo "############ dual-NUMA (no bind, both sockets) ############"
OMP_NUM_THREADS=64 python $WT/plugin/validate/throughput_bench.py $COMMON 2>&1 | grep -iE "prefill_tok|decode_tok"
echo "############ single-NUMA (numactl -N0 -m0, canonical per-rank) ############"
OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 \
  python $WT/plugin/validate/throughput_bench.py $COMMON 2>&1 | grep -iE "prefill_tok|decode_tok"
echo "[done numatie]"
