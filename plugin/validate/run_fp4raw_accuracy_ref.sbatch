#!/bin/bash
# Isolation: run the fp4-raw REFERENCE MoE (_torch_fp4_moe_apply: straightforward torch fp4->bf16
# dequant + SwiGLU from the same fp4 weights) to check whether garbage output is the MXFP4 KERNEL
# or a common path (attention/DSA). Coherent here => MXFP4-kernel bug; garbage => common bug.
# fp4-raw is slow (per-token dequant) so QUICK=1 (1 prompt) + few tokens.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_fp4raw_acc
#SBATCH --output=/scratch/bkaul/dsv4_fp4raw_acc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_FP4_MOE_BF16=1      # fp4-raw reference MoE (NOT the MXFP4 kernel)
unset INTEL_CPU_DSV4_MXFP4_MOE
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=3
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[exit=$?]"
