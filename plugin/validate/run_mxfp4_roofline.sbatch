#!/bin/bash
# Roofline microbench for the MXFP4 W4A16 CPU MoE kernel at the real DeepSeek-V4-Flash shape.
# Isolation perf gate (no model load); runs on any AMX node. Set BW_GBPS + NODE_LABEL per node.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --job-name=dsv4_mxfp4_roofline
#SBATCH --output=/scratch/bkaul/dsv4_mxfp4_roofline_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export NODE_LABEL=${SLURMD_NODENAME:-emr}
export THREADS=${THREADS:-8,16,32,43}
cd /data/nfs_home/bkaul
echo "=== DECODE (M=1) ==="
M=1  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/bench_mxfp4_moe_roofline.py
echo "=== PREFILL tile (M=64) ==="
M=64 python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/bench_mxfp4_moe_roofline.py
echo "[exit=$?]"
