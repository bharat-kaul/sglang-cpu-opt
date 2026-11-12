#!/bin/bash
#SBATCH --job-name=dsv4_meta_sink_simd
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --exclude=pcl-sprh04,pcl-sprh08,pcl-sprh09,pcl-sprh10
#SBATCH --output=/scratch/%u/dsv4_meta_sink_simd_%j.log
set -euo pipefail
ROOT=/data/nfs_home/bkaul/dsv4-standalone-meta-opt
PYTHON=/scratch/bkaul/venvs/sglang-cpu/bin/python
RESULTS="/scratch/bkaul/dsv4-standalone-meta-opt/sink-simd-job-${SLURM_JOB_ID}"
test ! -e "$RESULTS" || { echo "FATAL: results already exist"; exit 1; }
mkdir -p "$RESULTS/extensions"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=threads
export MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$RESULTS/extensions"
hostname > "$RESULTS/host.txt"
lscpu > "$RESULTS/topology.txt"
numactl --hardware > "$RESULTS/numa.txt"
for rep in 1 2 3; do
    echo "SINK_SIMD_START job=$SLURM_JOB_ID replicate=$rep"
    numactl --physcpubind=0-63 --membind=0 "$PYTHON" -u \
        plugin/validate/standalone_meta_sink_simd.py \
        --replicate "$rep" --output "$RESULTS/replicate-${rep}.json"
done
objdump -d -C "$RESULTS/extensions/meta_sink_simd4/meta_sink_simd4.so" > "$RESULTS/disassembly.txt"
echo "SINK_SIMD_COMPLETE job=$SLURM_JOB_ID results=$RESULTS"
