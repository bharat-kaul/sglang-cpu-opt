#!/bin/bash
# Build the PRIVATE sgl_kernel common_ops.so from the isolated source copy (aot_private) WITHOUT
# touching the shared venv's installed sgl_kernel. Extracts the .so and runs a baseline sanity/parity
# test. Persists the cmake build-dir (_skbuild) so later edits rebuild incrementally (~30s).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_kbuild
#SBATCH --output=/scratch/bkaul/dsv4_kbuild_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
PRIV=/data/nfs_home/bkaul/cpopt-bwprobe/aot_private
cd "$PRIV"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
pip wheel . --no-build-isolation -w dist --config-settings=build-dir=_skbuild 2>&1 | tail -45 || { echo "BUILD FAILED"; exit 1; }
WHL=$(ls -t dist/*.whl 2>/dev/null | head -1); echo "WHEEL=$WHL"
[ -z "$WHL" ] && { echo "NO WHEEL PRODUCED"; exit 1; }
rm -rf wheel_x; mkdir -p wheel_x; (cd wheel_x && unzip -o -q "../$WHL")
SO=$(find "$PRIV/wheel_x" -name "common_ops*.so" | head -1); echo "PRIVATE_SO=$SO"
[ -z "$SO" ] && { echo "NO .so IN WHEEL"; exit 1; }
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=128
cd /data/nfs_home/bkaul/cpopt-bwprobe/plugin/validate
echo "=== baseline sanity (gate off) — the PRIVATE unmodified build must load + run ==="
OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
  env SO_PATH="$SO" E=8 M=32 THREADS=$NODE0 SEED=0 ITERS=20 INTEL_CPU_DSV4_MOE_DBLBUF=0 \
      OUT=/scratch/bkaul/dsv4_moe_y_off.pt python test_private_kernel.py
echo "[exit=$?]"
