#!/bin/bash
# FUSION PROTOTYPE, move 1 (decisive + cheap): does keeping the thread pool HOT between ops speed up
# decode? That is EXACTLY the monolith's premise (persistent warm threads, no inter-op park/wake). Test it
# with a free env A/B BEFORE writing any fusion C++. Fresh process, 8-thread sticky operating point.
#   WARM=0 (baseline): OMP_WAIT_POLICY=passive + KMP_BLOCKTIME=0  -> park immediately between every op.
#   WARM=1 (warm pool): OMP_WAIT_POLICY=active + blocktime=infinite -> threads spin, stay hot (no park/wake).
# Run both:  for W in 0 1; do sbatch -p emr --export=ALL,WARM=$W --job-name dsv4_warm$W \
#               plugin/validate/run_dsv4_decode_warmpool.sbatch; done
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/bkaul/dsv4_warmpool_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
WARM=${WARM:-0}
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# locked operating point: 8-thread sticky decode cap (churn-free)
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
if [ "$WARM" = "1" ]; then
  export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite GOMP_SPINCOUNT=infinite
  TAG=warm
else
  export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0
  TAG=cold
fi
export CLEAN_CORES="8sticky/${TAG}"
set +x
RES=$WT/plugin/validate/results/dsv4_warmpool_${TAG}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" \
    --batches 1,32 --input-len 8 --output-len 6 --reps 6 --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[CLEAN\]|\[DECODE THREADCAP\]|decode_step|Error|Traceback|OOM" | tail -40
echo "############ RESULT ($TAG) ############"; cat "$RES" 2>/dev/null
echo "[done] warmpool $TAG job=$SLURM_JOB_ID"
