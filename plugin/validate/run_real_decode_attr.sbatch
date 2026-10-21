#!/bin/bash
# D2 go/no-go GATE: REAL-WEIGHT decode attribution (the dummy proxy misleads for thread/MoE decisions).
# Native MXFP4 experts (fits one GNR domain), tp=1, DECODE_THREADS=8 (the real-Flash thread-cap applied,
# so the MoE is at its ~0.1ms floor and the TRUE recoverable share shows). Phase-split TIMEIT -> decode
# op breakdown: does the ~40%-on-dummy torch DSA/MHC bucket (mla.attend loop + MHC + indexer) hold up on
# REAL weights, or does the MoE dominate? Uses the tree-pilot plugin (has the per-forward TIMEIT flush).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_realattr
#SBATCH --output=/scratch/bkaul/dsv4_realattr_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export INTEL_CPU_DSV4_DECODE_THREADS=8            # real-Flash MoE thread-cap (steep inverse scaling)
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0    # spin-wait fix (primary config lever)
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/$USER/dsv4_realattr_${SLURM_JOB_ID}.json
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size 32 --input-len 128 --output-len 8 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 8192 --swa-full-tokens-ratio 0.5 2>&1 \
  | grep -iE "DSV4 TIMEIT|overhead split|prefill|decode|latency|throughput|tok|moe|attn|dsa\.|mhc\.|%  "
echo "[exit=${PIPESTATUS[0]}] realattr"
