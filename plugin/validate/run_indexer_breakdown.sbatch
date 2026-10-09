#!/bin/bash
# Authoritative indexer_logits breakdown on the AMX target node (EMR): convert/pack/compute split per M,
# + the compressor 3-shape coverage. Drives the conversion-fusion optimization with real (AMX) numbers.
#SBATCH --job-name=idx_breakdown
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/idx_breakdown_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname)"

echo ">>> INDEXER_LOGITS BREAKDOWN (AMX target node, warm; convert/pack/compute per M)"
INDEXER_BREAKDOWN=1 numactl -N 0 -m 0 python - <<'PY'
import torch
from torch.utils.cpp_extension import load
il = load(name="il_bd", sources=["plugin/kernels/dsa_pilot/indexer_logits.cpp"],
          extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)
H, D, S = 64, 128, 1024
for M in (1, 8, 16, 32, 64):
    q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.rand(M, H)
    il.indexer_logits(q, kv, w)          # warm (fills cache / brgemm scratch)
    il.indexer_logits(q, kv, w)          # measured -> prints the breakdown line
PY

echo ">>> COMPRESSOR 3-SHAPE COVERAGE (R=128/D512, R=8/D512, R=8/D128)"
numactl -N 0 -m 0 python plugin/validate/bench_compressor.py
echo ">>> done"
