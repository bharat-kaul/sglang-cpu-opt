#!/bin/bash
# Per-kernel numeric parity gate for the native MXFP4 W4A16 CPU MoE kernel.
# Tiny synthetic test (no model load) — needs only an AMX node. Proves the fp4->bf16
# dequant + VNNI repack + fused_experts_cpu(MXFP4) matches an independent torch oracle.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --job-name=dsv4_mxfp4_parity
#SBATCH --output=/scratch/bkaul/dsv4_mxfp4_parity_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/test_mxfp4_moe_cpu.py
echo "[exit=$?]"
