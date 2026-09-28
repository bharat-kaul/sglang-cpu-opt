#!/bin/bash
# Isolation: fp8 Plan-A MoE (sglang's official fp4->fp8 cast + proven fp8 W8A16 CPU path) instead
# of the MXFP4 kernel. Coherent here => the bug is the MXFP4 real-ckpt wiring; still garbage =>
# the bug is NOT the MoE (projections / rope / norm / lm_head). Heavier (~275GB after fp4->fp8).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_fp8ref_acc
#SBATCH --output=/scratch/bkaul/dsv4_fp8ref_acc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
# NOTE: MXFP4_MOE intentionally UNSET -> default fp8 Plan-A dequant path.
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=12
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[exit=$?]"
