#!/bin/bash
# Layout-matched DRAM streaming microbench for the DSV4-Flash routed-expert MoE weight stream.
# Builds bw_probe.cpp on the EMR node (right ISA), binds to ONE NUMA domain (to match the 226 GB/s
# per-domain stream_triad roofline), and sweeps three access patterns at full-batch (256 experts) and
# an M=32-decode proxy (~160 activated experts). Separates "pattern ceiling" from "kernel overhead".
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --job-name=dsv4_bwprobe
#SBATCH --output=/scratch/bkaul/dsv4_bwprobe_%j.log
set -x
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
g++ -O3 -fopenmp -march=native bw_probe.cpp -o /tmp/bw_probe_$$ || { echo "BUILD FAILED"; exit 1; }
BIN=/tmp/bw_probe_$$
EB=$((12*1024*1024))   # 12 MB/expert (w13 8MB + w2 4MB, int8-packed MXFP4), matches real shapes
NE=256                 # DSV4-Flash: 256 routed experts
ACT_DECODE=${ACT:-160} # ~distinct experts activated at M=32 (32 tokens x top-6, with overlap)
ITERS=${ITERS:-8}
# One NUMA domain (the 226 GB/s figure is per-domain). Count node-0 CPUs for OMP threads.
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=64
echo "=== node0 cpus=$NODE0 · expert_bytes=$EB · full read=$(python3 -c "print(f'{$EB*$NE/1e9:.2f}')")GB ==="
run() { OMP_NUM_THREADS=$NODE0 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 \
          numactl --cpunodebind=0 --membind=0 "$BIN" "$EB" "$NE" "$1" "$2" "$ITERS"; }
echo "--- CEILING (one contiguous sweep = triad-equivalent of the SAME footprint) ---"
run $NE 2
echo "--- FULL expert sweep (256), contiguous order vs scattered order ---"
run $NE 0
run $NE 1
echo "--- M=32 decode proxy (~$ACT_DECODE activated experts), contiguous vs scattered ---"
run $ACT_DECODE 0
run $ACT_DECODE 1
rm -f "$BIN"
echo "[exit=$?]"
