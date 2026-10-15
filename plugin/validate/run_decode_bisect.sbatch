#!/bin/bash
# DECODE-focused per-op bisection: proxy (4-layer dummy) decode tok/s at B=32 with each authored C kernel
# toggled ONE AT A TIME (INTEL_CPU_DSV4_CK_<op>), vs all-OFF (torch) and all-ON. Localizes the in-engine
# regression: uniform across ops => runtime/thread pathology; one op => that kernel's in-engine call pattern.
# Worktree plugin (dsv4-tree-pilot). Decode is the priority work item.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --job-name=dsv4_decbisect
#SBATCH --output=/scratch/bkaul/dsv4_decbisect_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_decbisect_${SLURM_JOB_ID}
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
mkdir -p "$TORCH_EXTENSIONS_DIR"
MODEL=/scratch/bkaul/models/DeepSeek-V4-Flash
BENCH=$WT/plugin/validate/throughput_bench.py
cd /data/nfs_home/bkaul
run() { # $1=tag, remaining=CK_* env assignments
  local tag=$1; shift
  echo "###### DECODE B=32 CONFIG=$tag ######"
  env "$@" INTEL_CPU_DSV4_DECODE_THREADS=32 python "$BENCH" \
    --model "$MODEL" --config "$MODEL/config.json" --tp 1 \
    --batch 32 --input-len 128 --output-len 32 --layers 4 --load-format dummy --mem-fraction 0.85 \
    --out /scratch/bkaul/dsv4_decbisect_${SLURM_JOB_ID}_${tag}.json 2>&1 | tail -15
}
run off          INTEL_CPU_DSV4_DSA_CKERNEL=0
run indexer      INTEL_CPU_DSV4_DSA_CKERNEL=0 INTEL_CPU_DSV4_CK_INDEXER=1
run compressor   INTEL_CPU_DSV4_DSA_CKERNEL=0 INTEL_CPU_DSV4_CK_COMPRESSOR=1
run topk         INTEL_CPU_DSV4_DSA_CKERNEL=0 INTEL_CPU_DSV4_CK_TOPK=1
run sinkhorn     INTEL_CPU_DSV4_DSA_CKERNEL=0 INTEL_CPU_DSV4_CK_SINKHORN=1
run combine      INTEL_CPU_DSV4_DSA_CKERNEL=0 INTEL_CPU_DSV4_CK_COMBINE=1
run allon        INTEL_CPU_DSV4_DSA_CKERNEL=1
echo "[done decbisect]"
