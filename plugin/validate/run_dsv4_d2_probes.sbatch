#!/bin/bash
# D2 PRE-COMMIT PROBES: four cheap held-fixed checks on ONE GNR node (page-cache-amortized)
# that must pass BEFORE committing to the multi-week serving-integration wiring. Resolves the
# four under-verified claims from the adversarial self-audit:
#   (a) clean isolated real-weight M=32 decode  -> reps=3 median + spread resolves the 20-vs-46 s/step
#   (b) held-fixed M=32 thread A/B in ONE process-> real thread curve (control-file sweep, not cross-run)
#   (c) enumerate registered sgl_kernel:: CPU ops-> WIRE vs AUTHOR fork (DUMP_OPS, in-worker, authoritative)
#   (d) de-double-counted attribution w/ reps    -> firms the op shares (TIMEIT pass, separate load)
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_d2probes
#SBATCH --output=/scratch/bkaul/dsv4_d2probes_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
# KV-pool caps so the SNC-domain-bound MXFP4 load isn't OOM-killed (native ~1M ctx + SWA):
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=96
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
cd /data/nfs_home/bkaul

# bonus: best-effort system-wide data-movement counters (node is exclusive); never blocks.
perf stat -a -I 1000 -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
  -o /scratch/$USER/dsv4_d2probes_${J}_perfstat.log 2>/dev/null &
PERF_PID=$!

# ============ PASS 1 (CLEAN, no TIMEIT): probes (a)+(b)+(c) on one resident load ============
CTL=/scratch/$USER/dsv4_d2_decthreads_${J}.ctl; echo 8 > "$CTL"
export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL"
export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_d2_sweep_${J}.json       # (a)+(b)
export INTEL_CPU_DSV4_DUMP_OPS=$RDIR/dsv4_d2_sgl_ops_${J}.json       # (c)
set +x
stdbuf -oL -eL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32,1" --perf-thread-sweep "8,32,64,128" \
  --perf-reps 3 --perf-input 128 --perf-output 4 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2probes_${J}_pass1.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|DUMP_OPS|died|OOM|Error|Traceback"
echo "[pass1 exit=${PIPESTATUS[0]}] clean sweep + op-dump"

# ============ PASS 2 (TIMEIT): probe (d) de-double-counted attribution, reps, separate load ======
set -x
unset INTEL_CPU_DSV4_SWEEP_JSON INTEL_CPU_DSV4_DUMP_OPS
echo 8 > "$CTL"
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export INTEL_CPU_DSV4_TIMEIT_JSON=$RDIR/dsv4_d2_attribution_${J}.json
export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_d2_attr_sweep_${J}.json
set +x
stdbuf -oL -eL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" \
  --perf-reps 3 --perf-input 128 --perf-output 6 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2probes_${J}_pass2.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|DSV4 TIMEIT|phase|decode|died|OOM|Error|Traceback"
echo "[pass2 exit=${PIPESTATUS[0]}] TIMEIT attribution"

kill "$PERF_PID" 2>/dev/null
echo "[done] d2 probes job=$J"
