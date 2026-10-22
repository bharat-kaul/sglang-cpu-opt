#!/bin/bash
# Option 2 CPU leg: teacher-forced per-position logit capture on the frozen token IDs, REAL post-trained
# weights, all-C/C++ chain (DSA_CKERNEL=1). Clean compute node (login node is flaky under the Base
# download). Output -> compared offline vs the GPU capture for CPU-vs-GPU numerical fidelity.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_o2_cpu
#SBATCH --output=/scratch/bkaul/dsv4_o2_cpu_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DSA_CKERNEL=1   # all-C/C++ production chain
cd /data/nfs_home/bkaul
set +x
set -o pipefail
python -u $WT/plugin/validate/capture_logits.py \
  --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
  --device cpu --tp 1 --leg cpu_c --topk 5 --max-seqs 6 --mem-fraction 0.70 \
  --out $WT/plugin/validate/results/o2_logits_cpu_c.json 2>&1 \
  | grep -iE "capture|Error|Traceback|OOM|died"
RC=${PIPESTATUS[0]}
echo "[exit=${RC}] o2 cpu capture"
exit $RC
