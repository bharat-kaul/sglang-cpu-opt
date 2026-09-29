#!/bin/bash
# In-situ parity for the fp8 W8A16 dense projections (the last unverified Q/K-content op).
# Compares the fp8 kernel apply vs a torch block-dequant matmul on real weights. Prefill + 1 token.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_fp8_parity
#SBATCH --output=/scratch/bkaul/dsv4_fp8_parity_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_FP8_PARITY=1
export ATTN_DBG_FILE=/scratch/bkaul/dsv4_fp8_parity_${SLURM_JOB_ID}.txt
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=1
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "=== FP8 PARITY FILE ==="; cat "$ATTN_DBG_FILE" 2>/dev/null || echo "none"
echo "[exit=$?]"
