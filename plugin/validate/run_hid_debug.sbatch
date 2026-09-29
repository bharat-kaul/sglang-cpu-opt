#!/bin/bash
# Localize the shared-path bug: per-layer hidden-state stats (abs-mean/max/NaN) to a file.
# Prefill + 1 token. Reveals where the forward goes bad (NaN/explode/collapse).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=dsv4_hid_dbg
#SBATCH --output=/scratch/bkaul/dsv4_hid_dbg_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_HID_DEBUG=1
export HID_DBG_FILE=/scratch/bkaul/dsv4_hid_debug_${SLURM_JOB_ID}.txt
export INTEL_CPU_DSV4_ATTN_DEBUG=1
export ATTN_DBG_FILE=/scratch/bkaul/dsv4_attn_debug_${SLURM_JOB_ID}.txt
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_QUICK=1
export ACC_MAX_NEW=1
export ACC_PROMPT_IDX=0
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "=== HIDDEN-STATE DEBUG FILE ==="
cat "$HID_DBG_FILE" 2>/dev/null || echo "no hid debug file"
echo "[exit=$?]"
