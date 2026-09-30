#!/bin/bash
# Layer-truncated per-kernel profile (user's idea): run only the first N decoder layers so each kernel
# executes at its REAL shape but the forward is ~43/N cheaper -> a representative prefill input-len is
# affordable. num_hidden_layers is overridden via --json-model-override-args (no code change). Layers
# 0-3 cover all three attention types (compress_ratio 0=SWA, 4=C4-DSA, 128=C128-DSA) + MoE + hash-conn.
# INTEL_CPU_DSV4_TIMEIT=1 dumps the per-op [DSV4 TIMEIT] breakdown at exit. EMR=directional; GNR=roofline%.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_timeit_layers
#SBATCH --output=/scratch/bkaul/dsv4_timeit_layers_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_TIMEIT=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
LAYERS=${LAYERS:-4}
INLEN=${INLEN:-256}
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --json-model-override-args "{\"num_hidden_layers\": ${LAYERS}}" \
  --device cpu --tp 1 --batch-size 1 --input-len ${INLEN} --output-len 4 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 8192
echo "[exit=$?]  layers=${LAYERS} input-len=${INLEN}"
echo "=== per-op breakdown ==="
grep "DSV4 TIMEIT" /scratch/bkaul/dsv4_timeit_layers_${SLURM_JOB_ID}.log || true
