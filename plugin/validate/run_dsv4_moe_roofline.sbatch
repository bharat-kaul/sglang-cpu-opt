#!/bin/bash
# WAVE-3: MoE apply ROOFLINE at the warm operating point. Is the executed (vector) MoE path at its
# weight-streaming floor, or is there overhead headroom? Probe reports per decode-M: active experts, GB
# streamed, apply ms + GB/s, a packed-byte-gather floor ms + GB/s, and apply/floor ratio. Warm config,
# thread count passed via DTHR (default 32). Short workload (no timeout).
#   sbatch -p emr --export=ALL,DTHR=32 --job-name dsv4_moe_rl plugin/validate/run_dsv4_moe_roofline.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bkaul/dsv4_moe_rl_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
DTHR=${DTHR:-32}
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# warm + M-adaptive-matched fixed decode threads (sticky so the apply runs at DTHR)
export INTEL_CPU_DSV4_DECODE_THREADS=$DTHR INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
export INTEL_CPU_DSV4_MOE_ROOFLINE=1 INTEL_CPU_DSV4_MOE_OCC=1 \
       INTEL_CPU_DSV4_MOE_OCC_FILE=$WT/plugin/validate/results/moe_rl_occ_${SLURM_JOB_ID}.json
set +x
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" \
    --batches 1,8,32,64 --input-len 8 --output-len 6 --reps 2 --mem-fraction 0.80 2>&1 \
  | grep -iE "\[MOE ROOFLINE\]|\[CLEAN\]|Error|Traceback|OOM" | tail -40
echo "[done] moe roofline DTHR=$DTHR job=$SLURM_JOB_ID"
