#!/bin/bash
# DECISIVE precision test: force all fp8 dense projections to bf16 block-dequant on CPU.
# Coherent "Paris" => accumulated fp8 precision drift is the garbage cause; still garbage => logic bug.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_fp8bf16
#SBATCH --output=/scratch/bkaul/dsv4_fp8bf16_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_FP8_BF16=1
export INTEL_CPU_DSV4_KROPE_SHADOW=1
export ATTN_DBG_FILE=/scratch/bkaul/dsv4_fp8bf16_engage_${SLURM_JOB_ID}.txt
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=10
export ACC_PROMPT_IDX=0
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
