#!/bin/bash
# Indexer-logits fused q-prep A/B (bit-exact + streaming). EMR, 3 proc reps.
#SBATCH --job-name=idx_fused
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/idx_fused_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || { echo "FATAL: cannot resolve a git repository root"; exit 1; }
cd "$REPO"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 0 1 2; do
  echo ">>> ===== PROCESS REPLICATE $((p+1)) / 3 ====="
  PROC=$p numactl -N 0 -m 0 python plugin/validate/bench_indexer_fused.py
done
echo ">>> done"
