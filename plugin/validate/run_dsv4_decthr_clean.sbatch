#!/bin/bash
# CLEAN decode-thread confirm: fast 58-core load, then decode STICKY-capped to T (set once, no per-forward
# 58->T->58 churn), fresh process per T (no KV-pool-growth drift). Resolves BOTH resident-sweep confounds.
# Pass T via --export=ALL,CLEAN_T=<n>. Run T=2,4,8 as separate jobs:
#   for T in 2 4 8; do sbatch -p emr --export=ALL,CLEAN_T=$T --job-name dsv4_dthr_clean$T \
#       plugin/validate/run_dsv4_decthr_clean.sbatch; done
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/bkaul/dsv4_dthr_clean_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
T=${CLEAN_T:-8}
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# Fixed decode cap to T + STICKY (set once on first decode, never restored) -> zero per-forward churn.
# Load/prefill run uncapped on the full 58-core pin (fast load); only decode is capped.
export INTEL_CPU_DSV4_DECODE_THREADS=$T INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export CLEAN_CORES="load58/stickyT=$T"
set +x
RES=$WT/plugin/validate/results/dsv4_dthr_clean_T${T}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" \
    --batches 1,32 --input-len 8 --output-len 6 --reps 5 --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[CLEAN\]|\[DECODE THREADCAP\]|decode_step|cores|torch_threads|Error|Traceback|OOM" | tail -40
echo "############ RESULT ############"; cat "$RES" 2>/dev/null
echo "[done] clean decode-thread T=$T job=$SLURM_JOB_ID"
