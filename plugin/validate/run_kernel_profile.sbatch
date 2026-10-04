#!/bin/bash
# DSV4 fixed-point profile: split the real-decode 18ms (M=32,E=256, fused AVX GEMV) into DRAM-stream vs
# compute/latency floor by toggling INTEL_CPU_DSV4_MOE_PINEXPERT (pins all expert weight reads to expert 0
# -> SAME path+routing+m_size, but cache-resident). normal - pinned = DRAM component. Pinned near 18 => the
# path is compute/latency-bound (real fixed-point code headroom). Pinned near ~7 => bandwidth-bound (little).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_kprof
#SBATCH --output=/scratch/bkaul/dsv4_kprof_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
PRIV=/data/nfs_home/bkaul/cpopt-bwprobe/aot_private
cd "$PRIV"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
pip wheel . --no-build-isolation -w dist --config-settings=build-dir=_skbuild 2>&1 | tail -12 || { echo "BUILD FAILED"; exit 1; }
WHL=$(ls -t dist/*.whl | head -1); rm -rf wheel_x; mkdir -p wheel_x; (cd wheel_x && unzip -o -q "../$WHL")
SO=$(find "$PRIV/wheel_x" -name "common_ops*.so" | head -1); echo "PRIVATE_SO=$SO"
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=128
cd /data/nfs_home/bkaul/cpopt-bwprobe/plugin/validate
prof() { local E=$1 PIN=$2; local TAG="E=$E PIN=$PIN"; echo "======== $TAG ========"; \
  OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
    env SO_PATH="$SO" E=$E M=32 THREADS=$NODE0 SEED=0 ITERS=20 INTEL_CPU_DSV4_MOE_PATHLOG=1 \
        ${PIN:+INTEL_CPU_DSV4_MOE_PINEXPERT=1} python test_private_kernel.py 2>&1 \
    | grep -aE "DSV4_MOE_PATHLOG|PRIVATE-KERNEL"; }
# real decode point
prof 256 ""     # normal: full DRAM stream (~135 distinct experts)
prof 256 1      # pinned: cache-resident -> compute/latency floor (same AVX path, same m_size~1)
# second AVX point for a trend
prof 64 ""
prof 64 1
echo "[exit=$?]"
