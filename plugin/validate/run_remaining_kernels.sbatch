#!/bin/bash
# Phase-1 no-regression + correctness measurement for the remaining authored kernels on the EMR target
# node: indexer_topk (#4), sinkhorn (#5), combine (#6). Confirms the C/C++ floor vs torch across the
# M-sweep (set-match / cosine held) before the Phase-1 pause-for-review.
#SBATCH --job-name=rem_kernels
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/rem_kernels_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname)"

echo ">>> INDEXER_TOPK (#4) — set-match vs torch.topk + no-regression floor"
numactl -N 0 -m 0 python plugin/validate/bench_topk.py

echo ">>> SINKHORN (#5) — cosine vs torch oracle + no-regression floor (watch small-M dispatch overhead)"
numactl -N 0 -m 0 python plugin/validate/bench_sinkhorn.py

echo ">>> COMBINE (#6) — cosine vs torch oracle + no-regression floor (watch small-M dispatch overhead)"
numactl -N 0 -m 0 python plugin/validate/bench_combine.py
echo ">>> done"
