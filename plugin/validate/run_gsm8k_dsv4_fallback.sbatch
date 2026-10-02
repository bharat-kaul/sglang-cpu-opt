#!/bin/bash
# FALLBACK gsm8k for DeepSeek-V4-Flash: smaller subset + CHUNKED checkpointing harness.
# Use if the full 1319-Q single-shot run (task_gsm8k.py) walls out with no json.
# Writes a PARTIAL json after every chunk, so a wall/watchdog kill still yields a number.
# Override NQ / CHUNK / MAXNEW at submit: sbatch --export=ALL,NQ=330,CHUNK=64 run_gsm8k_dsv4_fallback.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_gsm8k_fb
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_fb_%j.log
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
python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-330} --chunk-size ${CHUNK:-64} --num-shots 8 \
    --max-new-tokens ${MAXNEW:-256} --watchdog-timeout 7000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_fb_${SLURM_JOB_ID}.json
echo "[exit=$?]"
