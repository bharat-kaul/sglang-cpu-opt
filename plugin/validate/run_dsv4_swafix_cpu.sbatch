#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --job-name=dsv4_swafix
#SBATCH --output=/scratch/bkaul/dsv4_swafix_%j.log
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DSA_CKERNEL=1
cd /data/nfs_home/bkaul
stdbuf -oL python -u $WT/plugin/validate/capture_logits.py \
  --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
  --device cpu --tp 1 --leg cpu_swafix --topk 5 --max-seqs 2 --mem-fraction 0.70 \
  --out $WT/plugin/validate/results/o2_logits_cpu_swafix.json 2>&1 | stdbuf -oL grep -iE "capture|fell back|Error|Traceback|OOM" | tail -6
echo "[exit=${PIPESTATUS[0]}]"
