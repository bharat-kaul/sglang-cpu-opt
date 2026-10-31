#!/bin/bash
# Verify decode thread optimum at S=128 (longer context -> more KV work -> maybe wants more threads than S=8's 32).
# Warm + coop, B=32, sweep fixed decode threads via sticky cap. T passed via CLEAN_T.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bkaul/dsv4_decthr_s128_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot; CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow; T=${CLEAN_T:-32}
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=$T INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000 INTEL_CPU_DSV4_COOP_MHC=1
set +x
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py --model "$CKPT" --config "$CKPT/config.json" \
    --batches 32 --input-len 128 --output-len 10 --reps 3 --mem-fraction 0.80 2>&1 \
  | grep -iE "\[CLEAN\]|Error|Traceback" | tail -4
echo "[done] decthr s128 T=$T job=$SLURM_JOB_ID"
