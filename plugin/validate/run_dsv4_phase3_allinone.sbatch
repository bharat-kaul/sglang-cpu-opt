#!/bin/bash
# ONE real-weight GNR load -> FOUR answers (high-information run):
#   1. D2 gate   : decode op attribution (TIMEIT phase-split, snapshot .perfprobe) at DECODE_THREADS=8
#   2. Phase-3 perf: real-weight B=32 prefill+decode tok/s (perf-probe) vs the dummy-proxy 91/9.4
#   3. data-movement: perf stat -a -I (IPC, LLC-miss, cache-miss) -> DRAM/cache vs compute (best-effort bg)
#   4. Phase-3 task: gsm8k accuracy (chunked, checkpointing, sneak-previewable)
# Native MXFP4 experts (fit one GNR domain), tp=1, spin-wait fix, thread-cap=8 (real-Flash optimum).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=dsv4_phase3
#SBATCH --output=/scratch/bkaul/dsv4_phase3_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export INTEL_CPU_DSV4_DECODE_THREADS=8          # real-Flash MoE thread-cap (steep inverse scaling)
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0  # spin-wait fix (primary config lever)
# KV-pool caps so the scheduler isn't OOM-killed on the SNC-domain-bound MXFP4 load (native ~1M ctx + SWA):
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=64
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/$USER/dsv4_phase3_${SLURM_JOB_ID}.json
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul

# (3) data-movement: best-effort system-wide perf counters at 1s intervals (node is exclusive).
#     Never blocks the run; if perf is restricted it just errors into its own log.
perf stat -a -I 1000 -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
  -o /scratch/$USER/dsv4_phase3_${SLURM_JOB_ID}_perfstat.log 2>/dev/null &
PERF_PID=$!

# (1)+(2)+(4): one load -> perf-probe (perf+attribution) THEN gsm8k accuracy (chunked, checkpointing).
python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-batch 32 --perf-input 128 --perf-output 8 \
  --num-questions 100 --num-shots 8 --chunk-size 32 --max-new-tokens 256 \
  --mem-fraction 0.70 --acc-min 0.30 \
  --out /scratch/$USER/dsv4_phase3_gsm8k_${SLURM_JOB_ID}.json \
  --gens-out /scratch/$USER/dsv4_phase3_gens_${SLURM_JOB_ID}.json 2>&1 \
  | tee /scratch/$USER/dsv4_phase3_${SLURM_JOB_ID}_full.log \
  | grep -iE "PERF-PROBE|DSV4 TIMEIT|overhead split|chunk |acc=|accuracy|verdict|prefill|decode|latency|Error|Traceback|OOM|died|%  "
RC=${PIPESTATUS[0]}
kill "$PERF_PID" 2>/dev/null
echo "[exit=$RC] phase3"
