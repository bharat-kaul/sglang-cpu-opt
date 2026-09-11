#!/usr/bin/env bash
# Establish the ACHIEVABLE roofline ceilings on the target node:
#   - AMX BF16 compute peak (per-core + single-SNC + single-socket)
#   - DRAM memory bandwidth (STREAM triad, single socket)
# Run on-node via srun. Emits JSON lines consumed by the roofline verdict.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcc -O3 -march=native -mamx-tile -mamx-bf16 -fopenmp "$D/amx_peak.c"    -o "$D/amx_peak"
gcc -O3 -march=native -fopenmp "$D/stream_triad.c" -o "$D/stream_triad"

export OMP_PROC_BIND=close OMP_PLACES=cores

echo "=== AMX compute peak: single core ==="
OMP_NUM_THREADS=1 numactl --physcpubind=0 --membind=0 "$D/amx_peak"

echo "=== AMX compute peak: single SNC domain (node 0, 42 cores) ==="
OMP_NUM_THREADS=42 numactl --cpunodebind=0 --membind=0 "$D/amx_peak"

echo "=== AMX compute peak: single socket (nodes 0-2, 128 cores) ==="
OMP_NUM_THREADS=128 numactl --cpunodebind=0,1,2 --membind=0,1,2 "$D/amx_peak"

echo "=== AMX compute peak: full node (256 cores) ==="
OMP_NUM_THREADS=256 "$D/amx_peak"

echo "=== Memory bandwidth: single socket (STREAM triad) ==="
OMP_NUM_THREADS=128 numactl --cpunodebind=0,1,2 --membind=0,1,2 "$D/stream_triad"

echo "=== Memory bandwidth: full node (STREAM triad, interleaved) ==="
OMP_NUM_THREADS=256 numactl --interleave=all "$D/stream_triad"
