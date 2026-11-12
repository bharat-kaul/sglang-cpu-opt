#!/bin/bash
# Reproduce prior mixed-precision work; this is not a new optimization or promotion gate.
#SBATCH --job-name=dsv4_meta_sparse
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclude=pcl-sprh04,pcl-sprh08,pcl-sprh09,pcl-sprh10
#SBATCH --output=/scratch/%u/dsv4_meta_sparse_%j.log
set -euo pipefail
ROOT=/data/nfs_home/bkaul/dsv4-standalone-meta-opt
PYTHON=/scratch/bkaul/venvs/sglang-cpu/bin/python
RESULTS="/scratch/bkaul/dsv4-standalone-meta-opt/sparse-job-${SLURM_JOB_ID}"
test ! -e "$RESULTS" || { echo "FATAL: results already exist"; exit 1; }
mkdir -p "$RESULTS/build" "$RESULTS/results"
cd "$ROOT"
test "$(git rev-parse HEAD)" = 76cbcff6e0bcd81d0320262e5e9f39e723144089
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$RESULTS/build"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=threads OMP_WAIT_POLICY=ACTIVE
KERNEL="$ROOT/plugin/kernels/dsa_pilot/sparse_attention_kernel"
git rev-parse HEAD > "$RESULTS/revision.txt"
sha256sum plugin/validate/run_standalone_meta_sparse_baseline.sbatch > "$RESULTS/launcher.sha256"
numactl --hardware > "$RESULTS/numa.txt"
echo "SPARSE_BASELINE_BUILD job=$SLURM_JOB_ID"
numactl --physcpubind=0-63 --membind=0 "$PYTHON" \
    "$KERNEL/bench_matched_allocations.py" --build "$RESULTS/build" --build-only
objdump -d "$RESULTS/build/libsparse_attn_emr.so" > "$RESULTS/core-disassembly.txt"
bash "$KERNEL/run_matched_allocations.sh" "$RESULTS/build" "$RESULTS/results" "$PYTHON"
echo "SPARSE_BASELINE_COMPLETE job=$SLURM_JOB_ID results=$RESULTS"
