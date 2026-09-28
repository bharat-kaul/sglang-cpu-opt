#!/bin/bash
# Per-prompt input-dependence probe: one prompt (ACC_PROMPT_IDX), 1 token, capture logits.
# Launch several in parallel (different PROMPT_IDX) to test whether the wrong token depends on input.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_prompt_probe
#SBATCH --output=/scratch/bkaul/dsv4_prompt_probe_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_HID_DEBUG=1
export HID_DBG_FILE=/scratch/bkaul/dsv4_prompt_probe_${SLURM_JOB_ID}.txt
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export ACC_PROMPT_IDX=${ACC_PROMPT_IDX:-0}
export ACC_MAX_NEW=1
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "=== LOGITS ==="; grep -a LOGITS "$HID_DBG_FILE" 2>/dev/null
echo "[exit=$?]"
