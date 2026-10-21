#!/bin/bash
# Re-tune the B=32 real-weight decode thread count. threads=8 (M=1 optimum) STARVES the real 256-expert
# MoE at B=32 (29% of decode) -> 1.59 tok/s. Sweep DECODE_THREADS, page-cache amortized (one load warm,
# rest fast). perf-only (skip gsm8k). Finds the max-throughput decode thread count on REAL weights.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_tsweep
#SBATCH --output=/scratch/bkaul/dsv4_tsweep_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=64
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
for DT in 8 16 32 64 128; do
  echo "############ DECODE_THREADS=$DT ############"
  set +x
  INTEL_CPU_DSV4_DECODE_THREADS=$DT python -u $WT/plugin/validate/task_gsm8k_chunked.py \
    --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
    --perf-probe --perf-only --perf-batch 32 --perf-input 128 --perf-output 6 \
    --num-questions 1 --mem-fraction 0.70 2>&1 | grep -iE "PERF-PROBE|died|OOM|Error"
  set -x
done
echo "[done tsweep]"
