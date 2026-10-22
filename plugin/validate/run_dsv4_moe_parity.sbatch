#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_moeparity
#SBATCH --output=/scratch/bkaul/dsv4_moeparity_%j.log
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_MOE_PARITY=1   # in-situ kernel-vs-fp4-oracle probe (first layer, real ckpt)
export ATTN_DBG_FILE=$WT/plugin/validate/results/moe_parity_insitu.txt
rm -f $ATTN_DBG_FILE
export MODEL_PATH=/scratch/bkaul/models/DeepSeek-V4-Flash MAX_NEW=1 LOG_LEVEL=warning
export SMOKE_PROMPTS="What is 12 plus 7?"
cd /data/nfs_home/bkaul
python -u $WT/plugin/validate/_smoke.py 2>&1 | grep -iE "MXFP4 PARITY|smoke OK|Error|Traceback" | tail -8
echo "=== in-situ parity file ==="; cat $ATTN_DBG_FILE 2>/dev/null
echo "[exit=${PIPESTATUS[0]}]"
