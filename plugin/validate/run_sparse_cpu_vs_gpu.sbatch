#!/bin/bash
# Part B: diff the compiled CPU sparse_attend paths (scalar/fp32bmm/amx/bestof) against the REAL TileLang
# kernel output (saved by the H200 oracle) on an AMX node, to decide whether bf16-AMX is conformant.
#SBATCH --job-name=sparse_cpu_vs_gpu
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/sparse_cpu_vs_gpu_%j.log
set -euo pipefail
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
echo "node=$(hostname)"
numactl -N 0 -m 0 python plugin/validate/test_sparse_cpu_vs_gpu.py
