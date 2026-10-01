#!/bin/bash
# Path A: in-model BATCHED decode throughput via the real sgl.Engine (full scheduler +
# continuous batching, unified_radix_cache) on the PERF-PROXY (4-layer dummy, no 149GB load).
# Confirms the decode-MoE batching win end-to-end. Sweeps batch with decode threads matched
# to the batch sweet-spot (M=1->8, M>=32->32). OMP passive wait (spin-wait fix) always on.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_throughput
#SBATCH --output=/scratch/bkaul/dsv4_throughput_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
MODEL=/scratch/bkaul/models/DeepSeek-V4-Flash
LAYERS=${LAYERS:-4}
INLEN=${INLEN:-128}
OUTLEN=${OUTLEN:-32}
BENCH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/throughput_bench.py
cd /data/nfs_home/bkaul
# batch:decode_threads pairs (threads matched to the batch sweet-spot from the MXFP4 roofline sweep)
for pair in "1:8" "8:8" "16:16" "32:32"; do
  B=${pair%:*}; T=${pair#*:}
  echo "###### BATCH=$B decode_threads=$T ######"
  INTEL_CPU_DSV4_DECODE_THREADS=$T python "$BENCH" \
    --model "$MODEL" --config "$MODEL/config.json" --tp 1 \
    --batch "$B" --input-len "$INLEN" --output-len "$OUTLEN" \
    --layers "$LAYERS" --load-format dummy --mem-fraction 0.85 \
    --out /scratch/bkaul/dsv4_throughput_${SLURM_JOB_ID}_b${B}.json
done
echo "[exit=$?] layers=$LAYERS inlen=$INLEN outlen=$OUTLEN"
