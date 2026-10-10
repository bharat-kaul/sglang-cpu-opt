#!/bin/bash
# Performance-vs-roofline M-sweep for ALL shipped kernels (current I1/C1 build) on the EMR target node.
# 3 process replicates; aggregate medians offline.
#SBATCH --job-name=roofline_sweep
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/roofline_sweep_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"   # R6-F3
[[ -n "$REPO" ]] || REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"   # R7-F2: canonical ROOT (rev-parse already validates; .git may be a worktree file)
[[ -n "$REPO" ]] || { echo "FATAL: cannot resolve a git repository root"; exit 1; }
cd "$REPO"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 1 2 3; do
  echo ">>> ===== PROCESS REPLICATE $p / 3 ====="
  ROOFLINE_PROC=$p ROOFLINE_JSON=/scratch/${USER}/roofline_sweep_${SLURM_JOB_ID}_p${p}.json numactl -N 0 -m 0 python plugin/validate/bench_roofline_sweep.py
done
echo ">>> done"
