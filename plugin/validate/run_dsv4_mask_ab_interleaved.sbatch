#!/bin/bash
# DRIFT-ROBUST interleaved A/B for the vectorized vs loop coverage mask. ONE engine load; alternates the
# mask path per prefill rep (toggling INTEL_CPU_DSV4_MASK_LOOP, read per-call) so thermal/run-order drift
# hits both arms equally. Supersedes the confounded 3-sequential-leg job (393502), whose end-to-end tok/s
# was unreliable (middle leg faster than first). Reports paired-median realized prefill speedup.
# Submit: sbatch -p emr --job-name dsv4_maskab plugin/validate/run_dsv4_mask_ab_interleaved.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --job-name=dsv4_maskab
#SBATCH --output=/scratch/bkaul/dsv4_maskab_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS   # optimized throughput config
set +x
python -u $WT/plugin/validate/mask_ab_interleaved.py \
  --model "$CKPT" --config "$CKPT/config.json" --batch 32 --input-len 128 --reps 12 \
  --mem-fraction 0.70 --out $WT/plugin/validate/results/dsv4_mask_ab_${SLURM_JOB_ID}.json 2>&1 \
  | grep -iE "MASK A/B|REALIZED|loop|vec|speedup|gain|Error|Traceback" | tail -25
echo "[done] mask A/B job=$SLURM_JOB_ID"
