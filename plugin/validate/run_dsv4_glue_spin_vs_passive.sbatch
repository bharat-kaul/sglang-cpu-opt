#!/bin/bash
# MONOLITHIC-KERNEL DE-RISK (step 0): is the ~29% per-layer decode glue fork-join THREAD-WAKEUP (a config
# fix, no kernel) or fusible COMPUTE/DISPATCH (needs the monolithic per-layer kernel)? With passive wait,
# every parallel region re-wakes parked threads; spin keeps them hot. Compare SPIN vs PASSIVE at the SAME
# single-socket 8 threads and read the TIMEIT glue residual (layer.total - Σ timed-op) at each.
#   - if spin collapses the glue -> it's wakeup = config lever (OMP_WAIT_POLICY/thread-pool), NOT a kernel.
#   - if glue persists under spin -> it's dispatch/compute = the monolithic per-layer kernel is confirmed.
# Single socket, physical, 8 threads, dummy+real-router. Two bench legs (wait policy is launch-fixed).
# Submit: sbatch -p emr --job-name dsv4_glue plugin/validate/run_dsv4_glue_spin_vs_passive.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --job-name=dsv4_glue
#SBATCH --output=/scratch/bkaul/dsv4_glue_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_REAL_ROUTER=1 INTEL_CPU_DSV4_DECODE_THREADS=8
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
set +x

run_leg () {  # $1=tag  $2=waitpolicy  $3=blocktime
  local tag=$1 wp=$2 bt=$3
  local JSON=/scratch/bkaul/dsv4_glue_${SLURM_JOB_ID}_${tag}.json
  echo "===== LEG $tag  OMP_WAIT_POLICY=$wp KMP_BLOCKTIME=$bt ====="
  OMP_WAIT_POLICY=$wp KMP_BLOCKTIME=$bt OMP_NUM_THREADS=8 \
  INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1 INTEL_CPU_DSV4_TIMEIT_JSON="$JSON" TIMEIT_BATCH=32 \
  numactl --physcpubind=0-57 --membind=0 \
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code --load-format dummy \
      --mem-fraction-static 0.70 --batch-size 32 --input-len 128 --output-len 10 2>&1 \
    | grep -iE "Decode\.|median" | tail -2
  python3 - "$JSON" "$tag" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(f"  [{sys.argv[2]}] no json: {e}"); sys.exit()
dec=d.get("phases",{}).get("decode",{}); ops=dec.get("ops",[]); wall=dec.get("wall_s") or 0
par={"moe","dsa.mla.attend"}; timed=sum(o["s"] for o in ops if o["op"] not in par)
glue=max(0.0,wall-timed)
print(f"  [{sys.argv[2]}] layer.total={wall:.2f}s  Σtimed={timed:.2f}s  GLUE={glue:.2f}s ({100*glue/wall:.0f}% of layer time)")
PY
}
run_leg passive passive 0
run_leg spin    active   200
echo "############ VERDICT ############"
echo "  compare GLUE% passive vs spin: if spin's GLUE% drops a lot -> fork-join WAKEUP (config lever);"
echo "  if GLUE% ~unchanged -> dispatch/compute -> monolithic per-layer kernel confirmed."
echo "[done] glue spin-vs-passive job=$SLURM_JOB_ID"
