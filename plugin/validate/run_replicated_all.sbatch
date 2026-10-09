#!/bin/bash
# Replicated paired-trial requalification of all 6 kernels (R2-F3/P1-F6): order-varied paired timing vs a
# SAME-CONTRACT fallback, median+spread, 3 process-level repeats, full run identity.
#SBATCH --job-name=replicated_all
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/%u/replicated_all_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$ROOT"
GITSHA=$(git rev-parse HEAD 2>/dev/null) || { echo "FATAL: $ROOT is not a git repo"; exit 1; }
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
DIRTY=$(git status --porcelain | wc -l)
echo "node=$(hostname) git=$GITSHA dirty=$DIRTY OMP=$OMP_NUM_THREADS/$OMP_PROC_BIND torch=$(python -c 'import torch;print(torch.__version__)')"
echo "affinity=$(python -c 'import os;print(sorted(os.sched_getaffinity(0))[:6],"...")')"
[ "$DIRTY" -eq 0 ] || echo "WARNING: working tree DIRTY ($DIRTY files) — built object may not match $GITSHA"
for rep in 1 2 3; do
  echo ">>> PROCESS-REPEAT $rep"
  numactl -N 0 -m 0 python plugin/validate/bench_replicated.py
done
echo ">>> done"
