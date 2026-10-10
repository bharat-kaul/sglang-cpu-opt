#!/bin/bash
# Standalone-optimization step-3 paired measurement (review b51d1eb: I3/C2/C3) on the EMR target node.
# 3 process replicates. I3 baseline = git 3ed5869 (post-I1/pre-I3). C2/C3 compared vs the shipped C1.
#SBATCH --job-name=standalone_opt3
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/standalone_opt3_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export I3_BASELINE_REF=3ed5869
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 1 2 3; do
  echo ">>> ===== PROCESS REPLICATE $p / 3 ====="
  numactl -N 0 -m 0 python plugin/validate/bench_standalone_opt2.py
done
echo ">>> done"
