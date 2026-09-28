#!/bin/bash
# Accuracy check: greedy real-prompt generation with the native MXFP4 W4A16 MoE (tp=1).
# Coherent English/code output => the MXFP4 weight/scale/SwiGLU wiring is correct end-to-end.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_mxfp4_acc
#SBATCH --output=/scratch/bkaul/dsv4_mxfp4_acc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[exit=$?]"
