#!/bin/bash
# Build + measure the env-gated high-ILP MXFP4 GEMV (INTEL_CPU_DSV4_MOE_GEMV_UNROLL) at the real decode
# fixed point. For E in {256,64}: run gate-OFF (true baseline) and gate-ON, save both outputs, report ms and
# the gate-ON vs gate-OFF cosine + max rel err (reassociated fp32 sum -> expect cos~1, rel~1e-6, NOT bit-exact).
# Also confirm gate-OFF is unchanged vs the saved baseline. node0/128thr.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_kilp
#SBATCH --output=/scratch/bkaul/dsv4_kilp_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
PRIV=/data/nfs_home/bkaul/cpopt-bwprobe/aot_private
cd "$PRIV"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
pip wheel . --no-build-isolation -w dist --config-settings=build-dir=_skbuild 2>&1 | tail -20 || { echo "BUILD FAILED"; exit 1; }
WHL=$(ls -t dist/*.whl | head -1); rm -rf wheel_x; mkdir -p wheel_x; (cd wheel_x && unzip -o -q "../$WHL")
SO=$(find "$PRIV/wheel_x" -name "common_ops*.so" | head -1); echo "PRIVATE_SO=$SO"
NODE0=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, '!/^#/ && $2==0{c++} END{print c+0}'); [ "${NODE0:-0}" -lt 1 ] && NODE0=128
cd /data/nfs_home/bkaul/cpopt-bwprobe/plugin/validate
runk() { local E=$1 UNROLL=$2 OUT=$3; OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
    env SO_PATH="$SO" E=$E M=32 THREADS=$NODE0 SEED=0 ITERS=25 INTEL_CPU_DSV4_MOE_PATHLOG=1 \
        ${UNROLL:+INTEL_CPU_DSV4_MOE_GEMV_UNROLL=1} OUT=$OUT python test_private_kernel.py 2>&1 \
    | grep -aE "DSV4_MOE_PATHLOG|PRIVATE-KERNEL"; }
for E in 256 64; do
  echo "########## E=$E gate-OFF (baseline) ##########"; runk $E ""  /scratch/bkaul/dsv4_ilp_E${E}_off.pt
  echo "########## E=$E gate-ON  (high-ILP) ##########"; runk $E 1   /scratch/bkaul/dsv4_ilp_E${E}_on.pt
done
echo "########## PARITY: gate-ON vs gate-OFF (reassociated sum -> cosine, not bit-exact) ##########"
python - <<'PY'
import torch
for E in (256, 64):
    a = torch.load(f"/scratch/bkaul/dsv4_ilp_E{E}_off.pt").float()
    b = torch.load(f"/scratch/bkaul/dsv4_ilp_E{E}_on.pt").float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).abs().max() / (a.abs().max() + 1e-30)).item()
    bit = torch.equal(a, b)
    print(f"PARITY E={E}: cos={cos:.8f} max_rel_err={rel:.3e} bit_identical={bit}")
PY
echo "[exit=$?]"
