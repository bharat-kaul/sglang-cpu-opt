#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_gsm8k_clamp
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_clamp_%A_%a.log
# Matched FULL-ACCURACY run for the MXFP4 SwiGLU-clamp correction (reviewer-approved launch review
# DSv4-clamp-fullacc-REVIEW.md). Baseline comparison config (passive, 8-thread, installed build, no
# shadow/coop) with ONLY the correction enabled: accumulator ON + clamp ON across ALL routed layers.
set -euo pipefail
source /scratch/$USER/venvs/sglang-cpu/bin/activate
# CLEAN checkout at 9320cd1 (NOT the dirty main tree):
WT=/data/nfs_home/bkaul/dsv4-clamp-accuracy-worker
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
# REQUIRED correction (review s1): retain accumulator AND enable clamp on all routed layers.
export INTEL_CPU_DSV4_CHUNK_ACCUM=1
export INTEL_CPU_DSV4_MXFP4_CLAMP=1
# REQUIRED private per-shard extension dir + bounded build (review s3):
export TORCH_EXTENSIONS_DIR="/scratch/bkaul/dsv4-clamp-fullacc-ext/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MAX_JOBS=2
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Baseline comparison config (unchanged): cold/passive wait, 8-thread decode cap, NO coop, INSTALLED build.
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 INTEL_CPU_DSV4_COOP_MHC=0

# --- provenance stamp per shard (review s3; harness env whitelist omits both flags) ---
STAMP="/scratch/bkaul/dsv4_clamp_stamp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
INFRA=$WT/plugin/intel_cpu_models/_dsv4_cpu_infra.py
CLAMP_PY=$WT/plugin/intel_cpu_models/_dsv4_moe_clamp.py
CLAMP_H=$WT/plugin/kernels/dsv4_moe_clamp.h
{
  echo "runtag=clamp_fullacc accum=1 clamp=1 clamp_scope=all_layers"
  echo "worktree=$WT"
  echo "commit=$(git -C $WT rev-parse HEAD)"
  echo "short=$(git -C $WT rev-parse --short HEAD)"
  echo "tracked_worktree_dirty=$(test -z "$(git -C $WT status --porcelain)" && echo NO || echo YES)"
  echo "infra_sha256=$(sha256sum $INFRA | cut -d' ' -f1)"
  echo "clamp_loader_sha256=$(sha256sum $CLAMP_PY | cut -d' ' -f1)"
  echo "clamp_epilogue_sha256=$(sha256sum $CLAMP_H | cut -d' ' -f1)"
  echo "sglang_build=$(python -c 'import sglang;print(sglang.__version__)')"
  echo "model=/scratch/bkaul/models/DeepSeek-V4-Flash"
  echo "data=/scratch/bkaul/gsm8k_test.jsonl"
  echo "job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=$(hostname)"
  echo "chunked_prefill=512 shots=8 maxnew=256 nq=1319 nshards=${SLURM_ARRAY_TASK_COUNT} chunk_size=32"
  echo "OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 threads=8 coop_mhc=0 shadow=NO"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR max_jobs=2"
} > "$STAMP"
cat "$STAMP"

cd $WT/plugin/validate
python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions 1319 --num-shards ${SLURM_ARRAY_TASK_COUNT} --shard-index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 32 --num-shots 8 --max-new-tokens 256 --watchdog-timeout 7000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_clamp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
    --gens-out /scratch/bkaul/dsv4_clamp_gens_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
    --gens-schema-v2
echo "[done] clamp full-accuracy shard ${SLURM_ARRAY_TASK_ID}"
