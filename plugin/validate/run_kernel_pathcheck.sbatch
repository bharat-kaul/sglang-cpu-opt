#!/bin/bash
# Rebuild private kernel + confirm WHICH GEMM path the MXFP4 MoE takes at the real decode point (M=32,E=256)
# vs the compute-probe point (M=32,E=8). Decisive: use_brgemm=M*topk/E>4 -> E=256 should be the fused AVX
# GEMV path (dequant already fused), E=8 the Btmp/AMX brgemm path. Settles whether the double-buffer is relevant.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_kpath
#SBATCH --output=/scratch/bkaul/dsv4_kpath_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
PRIV=/data/nfs_home/bkaul/cpopt-bwprobe/aot_private
cd "$PRIV"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
pip wheel . --no-build-isolation -w dist --config-settings=build-dir=_skbuild 2>&1 | tail -15 || { echo "BUILD FAILED"; exit 1; }
WHL=$(ls -t dist/*.whl | head -1); rm -rf wheel_x; mkdir -p wheel_x; (cd wheel_x && unzip -o -q "../$WHL")
SO=$(find "$PRIV/wheel_x" -name "common_ops*.so" | head -1); echo "PRIVATE_SO=$SO"
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=128
cd /data/nfs_home/bkaul/cpopt-bwprobe/plugin/validate
for E in 256 64 16 8; do
  echo "======== PATH CHECK E=$E (avg tokens/expert = 192/$E) ========"
  OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
    env SO_PATH="$SO" E=$E M=32 THREADS=$NODE0 SEED=0 ITERS=3 INTEL_CPU_DSV4_MOE_PATHLOG=1 \
        python test_private_kernel.py 2>&1 | grep -aE "DSV4_MOE_PATHLOG|PRIVATE-KERNEL"
done
echo "[exit=$?]"
