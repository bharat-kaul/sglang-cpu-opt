#!/bin/bash
# Historical six-kernel reproduction only; no model loads or shared library changes.
#SBATCH --job-name=dsv4_meta_baseline
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=pcl-sprh04,pcl-sprh08,pcl-sprh09,pcl-sprh10
#SBATCH --output=/scratch/%u/dsv4_meta_baseline_%j.log
set -euo pipefail
ROOT=/data/nfs_home/bkaul/dsv4-standalone-meta-opt
PYTHON=/scratch/bkaul/venvs/sglang-cpu/bin/python
RESULTS="/scratch/bkaul/dsv4-standalone-meta-opt/job-${SLURM_JOB_ID}"
test ! -e "$RESULTS" || { echo "FATAL: results already exist"; exit 1; }
mkdir -p "$RESULTS"
cd "$ROOT"
test "$(git rev-parse HEAD)" = 76cbcff6e0bcd81d0320262e5e9f39e723144089
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$RESULTS/extensions"
mkdir "$TORCH_EXTENSIONS_DIR"
hostname > "$RESULTS/host.txt"
lscpu > "$RESULTS/topology.txt"
numactl --hardware > "$RESULTS/numa.txt"
"$PYTHON" -c 'import torch; print(torch.__config__.show())' > "$RESULTS/torch-config.txt"
for rep in 1 2 3; do
    export META_REPLICATE="$rep" OP_OUT="$RESULTS/replicate-${rep}.json"
    echo "BASELINE_REPLICATE_START job=$SLURM_JOB_ID replicate=$rep"
    numactl -N0 -m0 "$PYTHON" -u plugin/validate/standalone_meta_baseline.py
    echo "BASELINE_REPLICATE_COMPLETE job=$SLURM_JOB_ID replicate=$rep"
done
echo "BASELINE_COMPLETE job=$SLURM_JOB_ID results=$RESULTS"
