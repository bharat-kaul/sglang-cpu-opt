#!/bin/bash
# DEBUG: capture the FULL traceback of the phase3 engine-load failure (minimal, no grep, no perf-probe).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_p3dbg
#SBATCH --output=/scratch/bkaul/dsv4_p3dbg_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export INTEL_CPU_DSV4_DECODE_THREADS=8 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
set +x  # from here, DON'T trace — let the raw python traceback through unfiltered
python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --num-questions 1 --num-shots 8 --chunk-size 1 --max-new-tokens 8 \
  --mem-fraction 0.85 --out /scratch/$USER/dsv4_p3dbg_${SLURM_JOB_ID}.json
echo "[exit=$?] p3dbg"
