#!/bin/bash
# Replicated indexer measurement (P1-F6): re-measure the CONFORMED bf16-stage indexer kernel across 3
# independent trials and record run provenance (thread count, affinity, build). Prints the per-trial
# sweep so medians + spread can be taken honestly (not a single sequential-average labeled "median").
#SBATCH --job-name=idx_replicated
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/idx_replicated_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$ROOT"
GITSHA=$(git rev-parse HEAD 2>/dev/null) || { echo "FATAL: $ROOT is not a git repo"; exit 1; }
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
DIRTY=$(git status --porcelain | wc -l)
echo "node=$(hostname)  git=$GITSHA  dirty_files=$DIRTY  OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND"
echo "nproc=$(nproc)  torch=$(python -c 'import torch;print(torch.__version__)')  affinity=$(python -c 'import os;print(sorted(os.sched_getaffinity(0))[:4],\"...\")')"
[ "$DIRTY" -eq 0 ] || echo "WARNING: working tree DIRTY ($DIRTY files) — built object may not match $GITSHA"
for t in 1 2 3; do
  echo ">>> TRIAL $t — indexer sweep (conformed bf16-stage epilogue) + torch no-regression"
  numactl -N 0 -m 0 python plugin/validate/bench_idx_logits.py
done
echo ">>> done"
