#!/bin/bash
# Broader accuracy validation: gsm8k exact-match on DeepSeek-V4-Flash CPU (native MXFP4 MoE, tp=1).
# Post wo_a fix, expect non-trivial accuracy (coherent reasoning), not 0.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_gsm8k
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/task_gsm8k.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions 12 --num-shots 8 --max-new-tokens 160 \
    --watchdog-timeout 3600 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_${SLURM_JOB_ID}.json
echo "[exit=$?]"
