#!/bin/bash
# Comprehensive full-tensor per-layer capture (prefill + decode) on CPU for offline cosine-per-layer.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_cpu_fullcap
#SBATCH --output=/scratch/bkaul/dsv4_cpu_fullcap_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_FULLCAP=1
export FULLCAP_FILE=/scratch/bkaul/dsv4_cpu_fullcap_${SLURM_JOB_ID}.pt
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=1
export ACC_PROMPT_IDX=0
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[fullcap] $FULLCAP_FILE"; ls -la "$FULLCAP_FILE" 2>/dev/null
