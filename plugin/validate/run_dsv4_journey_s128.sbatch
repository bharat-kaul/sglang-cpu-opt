#!/bin/bash
# JOURNEY stage measurement at B=32 S=128 (full 43L real weights) for the progressive-improvement chart.
# STAGE selects the lever config. Each stage at the SAME operating point -> clean ladder.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_journey
#SBATCH --output=/scratch/bkaul/dsv4_journey_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot; CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
STAGE=${STAGE:-warm8}
case $STAGE in
  warm8)   export INTEL_CPU_DSV4_DECODE_THREADS=8  INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1; export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000; export INTEL_CPU_DSV4_COOP_MHC=0 ;;  # +warm pool
  warmadapt) export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48" INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1; export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000; export INTEL_CPU_DSV4_COOP_MHC=0 ;;  # +adaptive threads
esac
set +x
RES=$WT/plugin/validate/results/dsv4_journey_${STAGE}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py --model "$CKPT" --config "$CKPT/config.json" \
    --batches 32 --input-len 128 --output-len 10 --reps 4 --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[CLEAN\]|Error|Traceback" | tail -4
echo "######### $STAGE #########"; cat "$RES" 2>/dev/null
echo "[done] journey $STAGE job=$SLURM_JOB_ID"
