#!/bin/bash
# WALL-TIME 2-POINT (+ decomposition): the velocity story the user asked for — baseline (all-torch-eager
# + naive dispatch) vs final (all-C certified + dispatch addressed), one headline speedup. Three legs on
# ONE node (page-cache-amortized), each a FRESH process (so the per-process OMP_WAIT_POLICY differs):
#   1. BASELINE  : DSA eager (CKERNEL=0) + NAIVE dispatch (no thread cap, default spin wait)
#   2. mid       : DSA eager + OPTIMIZED dispatch (thread cap + passive)  -> isolates the dispatch win
#   3. FINAL     : DSA all-C (CKERNEL=1, provenance-CERTIFIED) + optimized dispatch -> isolates kernels
# MoE = AMX in all three (held fixed; always wired, not part of D2). Request shape M=32 (matches the
# B=32 headline), S=512 prompt, G=4 (3 timed decode steps), reps=3 -> representative wall-time.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_walltime
#SBATCH --output=/scratch/bkaul/dsv4_walltime_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=96
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
cd /data/nfs_home/bkaul

run_leg () {  # $1=ckernel(0/1) $2=dispatch(naive/opt) $3=tag
  ( # subshell so per-leg env (esp. OMP_WAIT_POLICY, read once at libgomp load) is isolated per process
    export INTEL_CPU_DSV4_DSA_CKERNEL=$1
    if [ "$2" = "opt" ]; then
      export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=8
    else
      unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS   # NAIVE: default spin, no cap
    fi
    export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_walltime_${J}_${3}.json
    export INTEL_CPU_DSV4_PROV_OUT=$RDIR/dsv4_walltime_${J}_prov_${3}.json
    stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
      --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
      --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "0" \
      --perf-reps 3 --perf-input 512 --perf-output 4 --num-questions 1 --mem-fraction 0.70 2>&1 \
      | stdbuf -oL grep -iE "PERF-SWEEP|fell back|died|OOM|Error|Traceback"
    echo "[leg ${3} exit=${PIPESTATUS[0]}]"
    echo "--- provenance ${3} (FINAL must be all-C: eager==0) ---"
    python $WT/plugin/validate/check_dispatch_provenance.py $RDIR/dsv4_walltime_${J}_prov_${3}.json || true
  )
}

set +x
echo "########## LEG 1: BASELINE (eager + naive dispatch) ##########"; run_leg 0 naive baseline
echo "########## LEG 2: eager + optimized dispatch ##########";        run_leg 0 opt   eager_opt
echo "########## LEG 3: FINAL (all-C certified + optimized) ##########"; run_leg 1 opt  final
echo "[done] walltime journey job=$J"
