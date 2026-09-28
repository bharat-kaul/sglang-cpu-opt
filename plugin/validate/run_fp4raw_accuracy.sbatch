#!/bin/bash
# Isolation: fp4-raw reference MoE (INTEL_CPU_DSV4_FP4_MOE_BF16) — dequant each routed expert
# fp4->bf16 in torch (sglang's official cast) then bf16 SwiGLU. Independent of the MXFP4 kernel.
# Slow (~100s/tok) so only 3 tokens. Coherent => MXFP4 kernel wiring bug; garbage => not the MoE.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --job-name=dsv4_fp4raw_acc
#SBATCH --output=/scratch/bkaul/dsv4_fp4raw_acc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_FP4_MOE_BF16=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=3
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[exit=$?]"
