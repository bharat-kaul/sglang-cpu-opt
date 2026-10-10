#!/bin/bash
# Standalone-optimization paired measurement (review b51d1eb: I1/I2/C1) on the EMR target node.
# Runs the paired candidate-vs-baseline bench as 3 SEPARATE PROCESSES (process-level replication); aggregate
# the per-process medians offline. Baseline kernels are read from git 020f429 (pre-optimization).
#SBATCH --job-name=standalone_opt
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/standalone_opt_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"   # R6-F3
[[ -n "$REPO" ]] || REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"   # R7-F2: canonical ROOT (rev-parse already validates; .git may be a worktree file)
[[ -n "$REPO" ]] || { echo "FATAL: cannot resolve a git repository root"; exit 1; }
cd "$REPO"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export BASELINE_REF=020f429
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 1 2 3; do
  echo ">>> ===== PROCESS REPLICATE $p / 3 ====="
  numactl -N 0 -m 0 python plugin/validate/bench_standalone_opt.py
done
echo ">>> done"
