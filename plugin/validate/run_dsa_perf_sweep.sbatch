#!/bin/bash
# Phase-A implementation-review PERF sweep on the roofline's measured-reference node (DDR5-5600).
# Runs every authored-kernel microbench 3x (median-of-runs; run-to-run noise floor ~33% on this cluster),
# threads bound ONCE at init, one socket / one NUMA domain. Absolute cpp_ms feeds the off-roofline join.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=dsa_perf_sweep
#SBATCH --output=/scratch/bkaul/dsa_perf_sweep_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
REPO=/data/nfs_home/bkaul/dsv4-tree-pilot
export PYTHONPATH=$REPO/plugin
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
cd "$REPO"
echo "=== perf-sweep node=$(hostname) $(date -Is) ==="
for BENCH in bench_idx_logits.py bench_topk.py bench_compressor.py bench_sinkhorn.py bench_combine.py bench_sparse_attend.py; do
  for REP in 1 2 3; do
    echo ">>> BENCH=$BENCH REP=$REP"
    OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 python "plugin/validate/$BENCH"
    echo "[exit=$? bench=$BENCH rep=$REP]"
  done
done
echo "=== done $(date -Is) ==="
