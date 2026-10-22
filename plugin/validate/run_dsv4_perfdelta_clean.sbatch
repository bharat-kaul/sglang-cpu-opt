#!/bin/bash
# Clean held-fixed eager-vs-C decode delta: the prior runs died because resident repeated-measurement
# degrades decode. FIX: measure ck=0 (eager) and ck=1 (C) in SEPARATE fresh engine loads (two python
# invocations), each --perf-reps 3; page cache amortizes the 2nd load on the same node. Gives the honest
# decode tok/s delta + TIMEIT breakdown without the degradation confound.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_perfdelta
#SBATCH --output=/scratch/bkaul/dsv4_perfdelta_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=96
export INTEL_CPU_DSV4_DECODE_THREADS=8
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
cd /data/nfs_home/bkaul

run_leg () {  # $1 = ck value (0 eager / 1 C), $2 = tag
  export INTEL_CPU_DSV4_DSA_CKERNEL=$1
  export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_perfdelta_${J}_ck$1.json
  stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
    --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
    --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" \
    --perf-reps 3 --perf-input 128 --perf-output 4 --num-questions 1 --mem-fraction 0.70 2>&1 \
    | stdbuf -oL grep -iE "PERF-SWEEP|fell back|died|OOM|Error|Traceback"
  echo "[leg ck=$1 ($2) exit=${PIPESTATUS[0]}]"
}

set +x
echo "===== LEG A: eager (ck=0), fresh engine ====="; run_leg 0 eager
echo "===== LEG B: all-C (ck=1), fresh engine (page-cache-amortized reload) ====="; run_leg 1 C
echo "[done] perfdelta job=$J"
