#!/bin/bash
# Step 5: REAL DeepSeek-V4-Pro on the GNR node — FULL DSA sparse path + BF16 showcase.
# BF16 compute (fp4 experts kept RAW in RAM ~790GB, dequant->bf16 in the MoE forward; fits
# tp=4 ~197GB/rank < 248GB/NUMA) + real DSA (compressor/indexer/sparse, NOT bypassed).
# Goal: accuracy vs reference + measured perf. Slow (torch dequant, no AMX) — accuracy probe.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=dsv4_pro_dsa
#SBATCH --output=/scratch/bkaul/dsv4_pro_dsa_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1          # full DSA sparse path (compressor/indexer/sparse)
export INTEL_CPU_DSV4_FP4_MOE_BF16=1     # fp4-storage / bf16-compute MoE (fits the node)
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
CKPT=/scratch/savancha/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 4 --batch-size 1 --input-len 8 --output-len 4 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 16384 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
