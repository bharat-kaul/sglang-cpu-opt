#!/bin/bash
# Newly-authored C/C++ kernel vs its NON-C/C++ torch baseline (the op each kernel replaces), on EMR.
# 3 process replicates; oneDNN-donor kernels (indexer_logits, sparse) flagged vs pure-authored (compressor,
# topk, sinkhorn, combine) in the emitted JSON. Order-varied paired trials; torch refs are noisy -> medians.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=cpp_vs_torch
#SBATCH --output=/scratch/bkaul/cpp_vs_torch_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || { echo "FATAL: cannot resolve a git repository root"; exit 1; }
export PYTHONPATH="$REPO/plugin"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
cd "$REPO"
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for p in 1 2 3; do
  echo ">>> ===== PROCESS REPLICATE $p / 3 ====="
  REPLICATED_PROC=$p REPLICATED_JSON=/scratch/${USER}/cpp_vs_torch_${SLURM_JOB_ID}_p${p}.json \
    numactl --cpunodebind=0 --membind=0 python plugin/validate/bench_replicated.py
done
echo ">>> done"
