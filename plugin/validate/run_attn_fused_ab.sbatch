#!/bin/bash
# D2 cheap-first: held-fixed A/B of the EXISTING manual-fused-softmax path (ATTN_FUSED=1) vs the default
# eager batched MLA attention. Attention is WEIGHT-INDEPENDENT -> dummy 4L proxy (fast, no 149GB load).
# Measures the decode mla.b.softmax / dsa.mla.attend time via phase-split TIMEIT. B=32 S=256 (realistic nk).
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_attnfused
#SBATCH --output=/scratch/bkaul/dsv4_attnfused_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=8
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
cd /data/nfs_home/bkaul
for FUSED in 0 1; do
  echo "############ ATTN_FUSED=$FUSED ############"
  set +x
  ENVX=""; [ "$FUSED" = 1 ] && ENVX="INTEL_CPU_DSV4_ATTN_FUSED=1"
  env $ENVX INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/$USER/dsv4_attnfused_${SLURM_JOB_ID}_f${FUSED}.json \
    python -u $WT/plugin/validate/throughput_bench.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json \
    --tp 1 --batch 32 --input-len 256 --output-len 8 --layers 4 --load-format dummy --mem-fraction 0.85 2>&1 \
    | stdbuf -oL grep -iE "decode_tok|prefill_tok|died|OOM|Error"
  set -x
done
echo "[done attnfused]"
