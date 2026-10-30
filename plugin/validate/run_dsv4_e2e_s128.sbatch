#!/bin/bash
# OVERALL e2e wall-time: naive baseline vs fully-optimized (config + coop-MHC), decode+prefill, B=1/32/64.
# Two fresh processes per arm (pre-spawn env = valid). MODE=naive | opt.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/bkaul/dsv4_e2e_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
MODE=${MODE:-opt}
if [ "$MODE" = "naive" ]; then
  export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
  export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0
  export INTEL_CPU_DSV4_COOP_MHC=0
else
  export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48" INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
  export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
  export INTEL_CPU_DSV4_COOP_MHC=1
fi
set +x
RES=$WT/plugin/validate/results/dsv4_e2e_s128_${MODE}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" --batches 32 --input-len 128 --output-len 12 --reps 4 \
    --mem-fraction 0.80 --out "$RES" 2>&1 | grep -iE "\[CLEAN\]|fell back|Error|Traceback" | tail -8
echo "######### RESULT ($MODE) #########"; cat "$RES" 2>/dev/null
echo "[done] e2e $MODE job=$SLURM_JOB_ID"
