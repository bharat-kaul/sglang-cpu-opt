#!/bin/bash
# SwiGLU-clamp A/B: teacher-forced per-position logit capture on the frozen token IDs, REAL
# post-trained weights, via the TORCH FP4 MoE path (_torch_fp4_moe_apply) so the DeepSeek SwiGLU
# CLAMP (config swiglu_limit=10) can be applied correctly (routed experts). Toggle clamp with
# INTEL_CPU_DSV4_SWIGLU_CLAMP. Compare each leg's top-1 vs the valid GPU-Flash logits
# (o2_logits_gpu0520.json) -> does the clamp move CPU toward GPU (0.82 -> ~0.95)?
# Submit with: sbatch --export=ALL,CLAMP=1,TAG=clampon run_dsv4_swiglu_ab.sbatch
#          and sbatch --export=ALL,CLAMP=0,TAG=clampoff run_dsv4_swiglu_ab.sbatch
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_swiglu_ab
#SBATCH --output=/scratch/bkaul/dsv4_swiglu_ab_%j.log
set -x
: "${CLAMP:=1}"; : "${TAG:=clampon}"
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_FP4_MOE_BF16=1   # route routed-experts through the TORCH path (clamp-capable)
unset INTEL_CPU_DSV4_MXFP4_MOE
export INTEL_CPU_DSV4_SWIGLU_CLAMP=$CLAMP
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DSA_CKERNEL=1
cd /data/nfs_home/bkaul
set +x
set -o pipefail
python -u $WT/plugin/validate/capture_logits.py \
  --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
  --device cpu --tp 1 --leg cpu_$TAG --topk 5 --max-seqs 2 --mem-fraction 0.70 \
  --out $WT/plugin/validate/results/o2_logits_cpu_$TAG.json 2>&1 \
  | grep -iE "capture|clamp|fell back|Error|Traceback|OOM|died"
RC=${PIPESTATUS[0]}
echo "--- top-1 vs version-matched GPU-Flash logits ---"
python $WT/plugin/validate/compare_logits.py \
  --a gpu=$WT/plugin/validate/results/o2_logits_gpu0520.json \
  --b cpu=$WT/plugin/validate/results/o2_logits_cpu_$TAG.json \
  --agree-min 0.95 2>&1 | tail -20 || true
echo "[exit=${RC}] swiglu A/B ($TAG, clamp=$CLAMP)"
exit $RC
