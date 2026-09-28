#!/bin/bash
# DeepSeek-V4-Flash decode at tp=1 with NATIVE MXFP4 experts (W4A16 CPU AMX kernel).
# The checkpoint's routed experts are already MXFP4 (packed fp4 + e8m0 group-32 scales). Instead of
# Plan-A dequant fp4->fp8 (275GB, exceeds one 258GB SNC domain -> forced the tp>1/interleave mess),
# keep them 4-bit and run sgl_kernel.fused_experts_cpu(CPUQuantMethod.MXFP4) which dequants fp4->bf16
# FUSED in the AMX GEMM. Experts stay ~137GB -> the full model FITS ONE DOMAIN -> tp=1 -> the decode
# thread-cap (225x M=1 fix) applies -> fast. Mirrors sglang's Mxfp4MoEMethod CPU path.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_flash_mxfp4tp1
#SBATCH --output=/scratch/bkaul/dsv4_flash_mxfp4tp1_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1                 # native MXFP4 W4A16 experts (fits one domain)
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8            # real-Flash MoE has steep inverse thread scaling
                                                  # (~0.1ms@8 vs ~2s@40); MoE dominates the forward
export INTEL_CPU_DSV4_TIMEIT=1
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
# Per-node RSS sampler: confirm the MXFP4 model fits node0 (~150-170GB, no spill/OOM).
( for _ in $(seq 1 30); do sleep 30; p=$(pgrep -f 'bench_one_batch' | head -1); \
    [ -n "$p" ] && echo "[numastat $(date +%H:%M:%S) pid=$p]" && numastat -p "$p" 2>/dev/null | tail -3; \
  done ) &
SAMPLER=$!
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size 1 --input-len 1 --output-len 8 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 4096 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
kill "$SAMPLER" 2>/dev/null
