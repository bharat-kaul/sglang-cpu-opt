#!/bin/bash
# PERSISTENT M x threads sweep on ONE GNR load. LEAN (M={1,32}, G=4) + CHECKPOINTED to a durable json
# (survives wall-kill; unbuffered observable) -- prior run lost all output to grep-buffering + a time-kill.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_mtsweep
#SBATCH --output=/scratch/bkaul/dsv4_mtsweep_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=96
CTL=/scratch/$USER/dsv4_decthreads_${SLURM_JOB_ID}.ctl; echo 8 > "$CTL"
export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL"
export INTEL_CPU_DSV4_SWEEP_JSON=/scratch/$USER/dsv4_mtsweep_${SLURM_JOB_ID}.json
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
set +x
stdbuf -oL -eL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32,1" --perf-thread-sweep "128,96,64,32,16,8" \
  --perf-input 128 --perf-output 4 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_mtsweep_${SLURM_JOB_ID}_full.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|died|OOM|Error|Traceback"
echo "[exit=${PIPESTATUS[0]}] mtsweep"
