#!/bin/bash
# COMPUTE-side probe for the DSV4-Flash MXFP4 MoE: run the REAL fused_experts_cpu kernel (synthetic
# weights, no model load) to decide whether the 170 GB/s (vs the 235 GB/s streaming ceiling the bw_probe
# measured) is MEMORY-overlap-limited or COMPUTE/dequant-limited.
#   Sweep A (E=256, DRAM-bound, vary M): if GB/s stays ~flat across a wide M (arithmetic-intensity) range,
#           170 is the kernel's streaming ceiling (overlap/dequant), not compute.
#   Sweep B (M=32, shrink E -> cache-resident): if the effective rate blows past 235, the op is MEMORY-bound
#           (fix = prefetch/double-buffer); if it caps near ~235, it is COMPUTE/dequant-bound (fix = tiling/
#           fusion/int8). 1 NUMA domain (matches the 226/235 per-domain numbers).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_moe_cc
#SBATCH --output=/scratch/bkaul/dsv4_moe_compute_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export HIDDEN=4096 INTER=2048 TOPK=6 ITERS=${ITERS:-30} BW_GBPS=226
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=128
export THREADS=${THREADS:-$NODE0}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
bind() { numactl --cpunodebind=0 --membind=0 python bench_mxfp4_moe_roofline.py; }
echo "############ SWEEP A: E=256 (3GB, DRAM-bound) — vary M (arithmetic intensity). Flat GB/s => streaming ceiling, not compute. ############"
for M in 1 8 32 64 128 256 512; do E=256 M=$M bind; done
echo "############ SWEEP B: M=32, shrink E so weights go cache-resident. >>235 => memory-bound (prefetch); ~235 => compute/dequant-bound (tiling/fusion). ############"
for E in 128 64 32 16 8 4; do E=$E M=32 bind; done
echo "############ SWEEP C: cache-resident small-E at the REAL per-expert density (M scaled so ~1 token/expert) ############"
E=8  M=1  bind
E=16 M=3  bind
echo "[exit=$?]"
