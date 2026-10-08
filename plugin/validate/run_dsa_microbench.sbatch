#!/bin/bash
# Parallel Phase-A op microbench: compile the pilot C++ kernel + parity (cosine vs torch oracle) +
# speedup, for ONE op (BENCH env). Fan out one op per node. One socket, pinned once.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=dsa_microbench
#SBATCH --output=/scratch/bkaul/dsa_microbench_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
REPO=/data/nfs_home/bkaul/dsv4-tree-pilot
export PYTHONPATH=$REPO/plugin
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
cd "$REPO"
BENCH=${BENCH:-bench_compressor.py}
echo "=== microbench $BENCH ==="
OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 python "plugin/validate/$BENCH"
echo "[exit=$? bench=$BENCH]"
