#!/usr/bin/env bash
# Run the BF16 AMX GEMM sweep on a Granite Rapids Slurm node.
# Applies the skill's affinity/NUMA plan and verifies AMX dispatch via oneDNN.
# Intended to be launched on-node (via srun ... bash slurm_gemm.sh).
set -euo pipefail

VENV="${VENV:-/scratch/$USER/venvs/amx}"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$VENV/bin/activate"

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export KMP_AFFINITY="granularity=fine,compact,1,0"

M=${M:-8192}; N=${N:-8192}; K=${K:-8192}; ITERS=${ITERS:-50}

echo "=== node: $(hostname) ==="
echo "=== AMX all-core frequency probe (perf, 128-thread BF16 load) ==="
if command -v perf >/dev/null 2>&1; then
  OMP_NUM_THREADS=128 numactl --cpunodebind=0,1,2 --membind=0,1,2 \
    perf stat -e cycles,task-clock \
    python "$TOOLS/gemm_amx_bf16.py" --M $M --N $N --K $K --iters $ITERS \
    --threads 128 --tag freqprobe 2>&1 | \
    grep -E "cycles|task-clock|tflops" || true
else
  echo "perf unavailable; frequency will be inferred"
fi

echo "=== ISA dispatch check (ONEDNN_VERBOSE, expect avx512_core_amx) ==="
ONEDNN_VERBOSE=1 OMP_NUM_THREADS=32 numactl --cpunodebind=0 --membind=0 \
  python "$TOOLS/gemm_amx_bf16.py" --M 2048 --N 2048 --K 2048 --iters 1 --warmup 0 \
  2>&1 | grep -iE "amx|brgemm|isa" | head -3 || echo "(no verbose lines captured)"

echo "=== single-socket sweep (128 physical cores, NUMA 0-2) ==="
for T in 64 96 128; do
  OMP_NUM_THREADS=$T numactl --cpunodebind=0,1,2 --membind=0,1,2 \
    python "$TOOLS/gemm_amx_bf16.py" --M $M --N $N --K $K --iters $ITERS \
    --threads $T --check --tag "socket0_t${T}"
done

echo "=== full-node (256 physical cores, interleaved memory) ==="
OMP_NUM_THREADS=256 numactl --interleave=all \
  python "$TOOLS/gemm_amx_bf16.py" --M $M --N $N --K $K --iters $ITERS \
  --threads 256 --tag "fullnode_t256"
