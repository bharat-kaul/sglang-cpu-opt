#!/bin/bash
# Diagnostic: dump gsm8k generations (truncation vs parse vs wrong-reasoning) to explain
# the 61% vs published 90.8% gap. CPU-only, small (16 Q x 2 settings).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_gsm8k_diag
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_diag_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
cd /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate
python diag_gsm8k_dump.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --n ${N:-16} --num-shots 8 --chunked-prefill-size 512
echo "[exit=$?]"
