#!/bin/bash
# Incremental rebuild of the PRIVATE kernel + measure the env-gated prefetch-distance experiment.
# Sweeps INTEL_CPU_DSV4_MOE_PF_K in {64 (baseline), 256, 512} at E=256 (DRAM-bound, where prefetch can
# matter) and E=8 (cache-resident compute floor). Asserts BIT-PARITY across PF_K values (a prefetch hint
# is non-semantic -> identical output). No touch of the shared venv .so.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_kmeas
#SBATCH --output=/scratch/bkaul/dsv4_kmeas_%j.log
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
run() { local E=$1 PFK=$2 IT=$3; OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
    env SO_PATH="$SO" E=$E M=32 THREADS=$NODE0 SEED=0 ITERS=$IT INTEL_CPU_DSV4_MOE_PF_K=$PFK \
        OUT=/scratch/bkaul/dsv4_y_E${E}_pf${PFK}.pt python test_private_kernel.py; }
echo "############ E=8 (cache-resident compute floor) ############"
for PFK in 64 256 512; do run 8 $PFK 20; done
echo "############ E=256 (DRAM-bound — prefetch can matter here) ############"
for PFK in 64 256 512; do run 256 $PFK 12; done
echo "############ BIT-PARITY: PF_K must not change the output (prefetch is non-semantic) ############"
python - <<'PY'
import torch
for E in (8, 256):
    base = torch.load(f"/scratch/bkaul/dsv4_y_E{E}_pf64.pt")
    for pf in (256, 512):
        y = torch.load(f"/scratch/bkaul/dsv4_y_E{E}_pf{pf}.pt")
        ident = torch.equal(base, y)
        md = (base - y).abs().max().item()
        print(f"PARITY E={E} pf64 vs pf{pf}: bit_identical={ident} max_abs_diff={md}")
PY
echo "[exit=$?]"
