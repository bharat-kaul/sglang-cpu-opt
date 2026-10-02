#!/bin/bash
# TIME-ATTRIBUTION profile for the pivot chart: per-op wall-time split by PHASE (pf/dec) on real
# DeepSeek-V4-Flash, at realistic serving batch, with the SPIN-WAIT FIX ON (passive) so the split is
# real and not an OpenMP-contention artifact. Short workload (one prefill + a few decode steps).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_pivot
#SBATCH --output=/scratch/bkaul/dsv4_pivot_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_TIMEIT=1
export INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
# Engine path (bench_one_batch hits an sglang TreeCache 'evict_for_alloc' bug at batch>1 on CPU).
# One batch-BS chunk = batch-BS prefill (long 8-shot) + OLEN decode steps; the per-op TIMEIT dump
# (pf/dec split) is emitted at process exit via atexit -> lands in THIS log.
cd /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate
python task_gsm8k_chunked.py \
  --model "$CKPT" \
  --data /scratch/bkaul/gsm8k_test.jsonl \
  --num-questions ${BS:-32} --chunk-size ${BS:-32} --num-shots 8 \
  --max-new-tokens ${OLEN:-16} --watchdog-timeout 2000 --chunked-prefill-size 512
echo "[exit=$?]  (TIMEIT per-op pf/dec dump emitted above at process exit)"
