#!/bin/bash
# Baseline-vs-optimized x {4L,8L} wall-time matrix for ONE node (EMR or GNR), on current accuracy-correct
# HEAD. Each leg = a FRESH engine load (separate python invocation) to avoid resident-degradation confound.
# Real-router guard ON in all legs (dummy expert weights route like the real model). Two depths -> the
# two-depth linear model extrapolates prefill/decode to 43L; the matched-depth ratio is the robust speedup.
#
# Submit with the partition for the node, e.g.:
#   sbatch -p emr   --job-name dsv4_mtx_emr plugin/validate/run_dsv4_perf_matrix.sbatch
#   sbatch -p gnrap --job-name dsv4_mtx_gnr plugin/validate/run_dsv4_perf_matrix.sbatch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/bkaul/dsv4_perfmatrix_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
# correctness/capacity levers (ON in BOTH legs — not perf optimizations)
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_REAL_ROUTER=1          # mode-collapse guard (route like the real model)
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
cd /data/nfs_home/bkaul
set +x

run_leg () {  # $1=mode(baseline|optimized)  $2=layers
  local mode=$1 NL=$2
  ( # subshell so per-leg env doesn't leak across legs
    if [ "$mode" = "optimized" ]; then
      # B=32 THROUGHPUT-optimal (empirically attributed, job 392780): all C kernels + SPIN OMP + ALL cores.
      # NB: OMP passive REGRESSES throughput here (prefill 2.18x->1.59x, decode 1.4x->0.3x) — it is an
      # idle-thread-thrash lever, not a throughput lever; and the decode thread-cap=8 is a B=1 latency lever.
      export INTEL_CPU_DSV4_DSA_CKERNEL=1
      unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
    else
      # FULLY NAIVE: eager DSA, OMP spin-wait (default), no decode thread-cap
      export INTEL_CPU_DSV4_DSA_CKERNEL=0
      unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
    fi
    export INTEL_CPU_DSV4_PROV_OUT=$RDIR/perfmatrix_${J}_${NODE}_${mode}_${NL}L_prov.json
    echo "===== LEG mode=$mode layers=$NL node=$NODE ====="
    stdbuf -oL python -u $WT/plugin/validate/throughput_bench.py \
      --model "$CKPT" --config "$CKPT/config.json" --tp 1 \
      --batch 32 --input-len 128 --output-len 8 --layers $NL \
      --load-format dummy --mem-fraction 0.70 \
      --out $RDIR/perfmatrix_${J}_${NODE}_${mode}_${NL}L.json 2>&1 \
      | grep -iE "prefill_tok|decode_tok|REAL_ROUTER|injected|fell back|Error|Traceback|OOM" | tail -20
    echo "--- dispatch-provenance ($mode: optimized must be all-C / baseline all-eager) ---"
    python $WT/plugin/validate/check_dispatch_provenance.py $RDIR/perfmatrix_${J}_${NODE}_${mode}_${NL}L_prov.json 2>/dev/null || true
  )
}

for NL in 4 8; do
  run_leg baseline  $NL
  run_leg optimized $NL
done
echo "[done] perf matrix node=$NODE job=$J"
