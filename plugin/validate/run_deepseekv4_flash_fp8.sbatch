#!/bin/bash
# DeepSeek-V4-Flash (284B / 13B active) enablement — reusability + perf showcase.
# Flash FITS in fp8 (~285GB -> ~71GB/rank at tp=4), so use the FAST Plan-A fp8 AMX path
# (fp4->fp8 at load), which is bf16-compute-equivalent accuracy (fp8 represents the fp4
# levels exactly; bf16 activations). Full DSA sparse path. Same plugin as Pro — zero new
# kernels (day-0 reusability proof). This is the fp8 rung; bf16/int8 rungs are separate.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_flash_fp8
#SBATCH --output=/scratch/bkaul/dsv4_flash_fp8_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1          # full DSA sparse path
unset INTEL_CPU_DSV4_FP4_MOE_BF16        # fp8 fits -> fast AMX (not the slow bf16-in-forward)
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=auto  # decode-wide thread cap (bound-2) — the 200x M=1 fix
export INTEL_CPU_DSV4_TIMEIT=1             # per-op wall-time breakdown (kernel/torch/framework)
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 4 --batch-size 1 --input-len 1 --output-len 16 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 16384 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
