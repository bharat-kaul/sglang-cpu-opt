#!/bin/bash
# MoE decode DIAGNOSTIC: confirm MXFP4/AMX path + active-group rows + thread scaling at B=1 & B=32.
# Full physical cores, NO decode cap (so [MOE ISO] spans 4..58 threads). Submit:
#   sbatch -p emr --job-name dsv4_moe_diag plugin/validate/run_dsv4_moe_diag.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_moe_diag
#SBATCH --output=/scratch/bkaul/dsv4_moe_diag_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_MOE_ISO=1   # trigger [MOE DIAG] + [MOE ISO]
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
set +x
# full physical socket-0 cores (0-57), NO decode-threads cap -> ISO sweep sees full range
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/moe_decode_diag.py \
    --model "$CKPT" --config "$CKPT/config.json" --input-len 8 --output-len 4 --mem-fraction 0.80 2>&1 \
  | grep -iE "\[MOE DIAG\]|\[MOE ISO\]|\[DIAG\]|Error|Traceback|OOM|assert" | tail -60
echo "[done] moe decode diagnostic job=$SLURM_JOB_ID"
