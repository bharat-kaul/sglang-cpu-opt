#!/bin/bash
# P1 GPU REFERENCE leg: run the FROZEN token-ID set through NATIVE sglang CUDA (no plugin, no CPU env)
# -> the trusted reference for the paired GPU/CPU-eager/CPU-C equivalence (GSM8K review P1).
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_p1_gpu
#SBATCH --output=/scratch/bkaul/dsv4_p1_gpu_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
# NATIVE GPU: no SGLANG_USE_CPU_ENGINE, no SGLANG_EXTERNAL_MODEL_PACKAGE, no INTEL_CPU_* env.
export PYTHONPATH=$WT/plugin
export MAX_TOTAL_TOKENS=8192 SWA_RATIO=0.8
cd /data/nfs_home/bkaul
set +x
set -o pipefail
python -u $WT/plugin/validate/paired_eval.py \
  --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
  --device cuda --tp 4 --leg gpu --max-new 512 --mem-fraction 0.85 \
  --out-prefix $WT/plugin/validate/results/p1_paired 2>&1 \
  | grep -iE "paired|Error|Traceback|OOM|died"
RC=${PIPESTATUS[0]}
echo "[exit=${RC}] p1 gpu reference leg"
exit $RC
