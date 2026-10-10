#!/bin/bash
# S1 sparse blockwise-BF16 paired measurement (review b51d1eb) on the EMR target node. 3 process replicates.
# S1 vs the shipped fp32-bmm donor and best-of, across M x K. EXPERIMENTAL: a measured loss is a valid result.
#SBATCH --job-name=sparse_s1
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/sparse_s1_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"   # R6-F3
[[ -n "$REPO" && -d "$REPO/.git" ]] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
git -C "$REPO" rev-parse --show-toplevel >/dev/null 2>&1 || { echo "FATAL: cannot resolve a git repository (REPO=$REPO)"; exit 1; }
cd "$REPO"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 1 2 3; do
  echo ">>> ===== PROCESS REPLICATE $p / 3 ====="
  numactl -N 0 -m 0 python plugin/validate/bench_sparse_s1.py
done
echo ">>> done"
