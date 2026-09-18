#!/bin/bash
# DeepSeek-V4-Flash decode at tp=1 with FP4-RAW experts (no fp8 dequant, no NUMA interleave).
# Rationale: the fp8-dequant path (~275GB) exceeds one 258GB NUMA domain, and spreading a tp=1
# rank across domains is impossible to LOAD reliably (numa_set_interleave_mask only affects the
# calling thread; the dequant/prepack worker threads fault on node0; mbind(MPOL_MF_MOVE) does NOT
# migrate the already-faulted THP torch pages -- proven by test_mbind_migrate.py). So keep experts
# as FP4 (~149GB, INTEL_CPU_DSV4_FP4_MOE_BF16=1): total ~160-170GB FITS one domain on node0, needs
# NO interleave/mbind/TP, is a clean tp=1 showcase, and the decode thread cap (200x M=1 fix) works.
# MoE forward dequants fp4->bf16 per-expert (only the top-6 active experts/token at decode).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_flash_fp4tp1
#SBATCH --output=/scratch/bkaul/dsv4_flash_fp4tp1_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_FP4_MOE_BF16=1              # keep experts fp4 -> fits one domain, no interleave
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=auto         # tp=1 -> per-forward cap works (200x M=1 fix)
export INTEL_CPU_DSV4_TIMEIT=1
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
# Per-node RSS sampler: confirm the fp4-raw model fits node0 (no spill / no OOM).
( for _ in $(seq 1 30); do sleep 30; p=$(pgrep -f 'bench_one_batch' | head -1); \
    [ -n "$p" ] && echo "[numastat $(date +%H:%M:%S) pid=$p]" && numastat -p "$p" 2>/dev/null | tail -3; \
  done ) &
SAMPLER=$!
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size 1 --input-len 1 --output-len 16 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 4096 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
kill "$SAMPLER" 2>/dev/null
