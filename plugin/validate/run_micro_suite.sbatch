#!/bin/bash
# Isolation microbench suite (no model load): roofline/speedup for the secondary hotspots surfaced by
# the per-op profile - DSA indexer, DSA compressor, MLA gather, dense GEMM threads, MHC. Runs fast on
# one AMX node (seconds each). The expensive model-load profiles (attention sub-timer, etc.) get their
# own nodes; per the fan-out rule the whole hotspot list is investigated in parallel.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_micro_suite
#SBATCH --output=/scratch/bkaul/dsv4_micro_suite_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export BW_GBPS=${BW_GBPS:-150}
cd /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate
for B in bench_indexer_logits.py bench_mla_gather.py bench_dense_gemm_threads.py bench_mhc.py; do
    echo "======== $B ========"
    python "$B" 2>&1 | tail -30 || echo "[$B failed]"
done
echo "[micro-suite exit=$?]"
