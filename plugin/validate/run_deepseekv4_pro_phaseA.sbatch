#!/bin/bash
# Phase A: REAL DeepSeek-V4-Pro (806GB fp8) on the GNR node, DSA BYPASSED.
# Goal: validate real fp8 load + MHC/HashTopK with real hash tables + expert AMX fp8
# (resolves the dummy-weight packing artifacts) + backbone perf at true scale.
# NOTE: DSA bypassed => DENSE attention => this is NOT accuracy-parity (Phase B wires DSA).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_pro_A
#SBATCH --output=/scratch/bkaul/dsv4_pro_phaseA_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_BYPASS_DSA=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
CKPT=/scratch/savancha/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1
cd /data/nfs_home/bkaul
# SGLang CPU shards weights across NUMA via TP (1 rank/NUMA). 6 NUMA nodes, but
# num_attention_heads=128 is not divisible by 6, so use tp=4 (~201GB/rank < 248GB/NUMA).
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 4 --batch-size 1 --input-len 8 --output-len 4 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.90 --max-total-tokens 16384 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
