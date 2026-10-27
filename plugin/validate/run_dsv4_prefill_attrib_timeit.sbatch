#!/bin/bash
# RELIABLE prefill op-class attribution (WALL-CLOCK, categorized) at the OPTIMIZED THROUGHPUT operating point.
# Purpose: validate the self_cpu-profiler split (40% GEMM / 26% glue / 24% eager / 10% fallback-mm) with a
# WALL-CLOCK, category-tagged measurement BEFORE trusting the achievable-roofline / recovery premise.
# self_cpu over-weights many-call small ops (the known gotcha); TIMEIT times the actual op region and tags it
# kernel (modeled compute) / framework (glue: gather/mask/cvt/select) / torch (eager: dsa.mla.attend, MHC).
#
# Reliability controls:
#  - REAL weights, full 43L, REAL routing (no dummy, no mode-collapse, no depth extrapolation).
#  - bench_one_batch runs the forward IN-PROCESS so the plugin's atexit TIMEIT dump fires (sgl.Engine kills its
#    worker and swallows the dump = the documented blocker).
#  - OPTIMIZED THROUGHPUT config: DSA C kernels ON, SPIN OMP + all cores (NOT the passive/cap-8 latency config),
#    so the split explains the published 174.5 tok/s run, not a different regime.
#  - PHASE-split (pf vs dec) so we read the PREFILL columns.
#  - 3 reps (first = warm/compile discard); report the phase wall + per-category split.
# Submit: sbatch -p emr --job-name dsv4_pfattr plugin/validate/run_dsv4_prefill_attrib_timeit.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:10:00
#SBATCH --job-name=dsv4_pfattr
#SBATCH --output=/scratch/bkaul/dsv4_pfattr_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_DSA_CKERNEL=1          # optimized: all-C/AMX DSA path
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# THROUGHPUT config: spin OMP, all cores (do NOT set passive / decode-cap — that's the latency regime)
unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
JSON=/scratch/bkaul/dsv4_pfattr_${SLURM_JOB_ID}.json
export INTEL_CPU_DSV4_TIMEIT_JSON=$JSON
export TIMEIT_MODEL="DeepSeek-V4-Flash" TIMEIT_NODE="EMR 1-socket throughput (spin, all cores)" TIMEIT_BATCH=32
set +x

echo "############ prefill attribution: B=32 S=128, optimized throughput config ############"
python -m sglang.benchmark.one_batch \
  --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
  --mem-fraction-static 0.70 \
  --batch-size 32 --input-len 128 --output-len 8
echo "[exit=$?] TIMEIT JSON -> $JSON"

echo "############ per-category PREFILL split (kernel=compute / framework=glue / torch=eager / OTHER=untimed glue) ############"
python3 - "$JSON" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ph=d.get("phases",{})
if "prefill" not in ph:
    print("phases found:", list(ph)); sys.exit()
pf=ph["prefill"]; ops=pf["ops"]; wall=pf.get("wall_s") or 0.0
bucket={}
for o in ops:
    b=o.get("bucket","?"); bucket[b]=bucket.get(b,0.0)+o.get("s",0.0)
timed=sum(bucket.values())
other=max(0.0, wall-timed)
print(f"PREFILL wall(parent layer.total sum)={wall:.3f}s  timed-leaves={timed:.3f}s  OTHER(untimed glue)={other:.3f}s")
rows=list(bucket.items())+[("OTHER(untimed)",other)]
denom=wall if wall>0 else timed
for b,v in sorted(rows,key=lambda x:-x[1]):
    print(f"  {b:16} {v:7.3f}s  {100*v/denom:5.1f}% of wall")
# recovery-relevant rollup
compute=bucket.get("kernel",0.0)
overhead=bucket.get("framework",0.0)+bucket.get("torch",0.0)+other
print(f"\n  COMPUTE (kernel) = {100*compute/denom:.1f}%  |  RECOVERABLE OVERHEAD (framework+torch+untimed) = {100*overhead/denom:.1f}%")
print("  (compare to the self_cpu-profiler estimate: ~40% compute / ~60% overhead)")
print("\n--- top prefill leaves by time ---")
for o in sorted(ops,key=lambda x:-x.get('s',0))[:18]:
    print(f"  {o.get('op','?'):24} {o.get('bucket','?'):10} {o.get('s',0):7.3f}s")
PY
echo "[done] prefill attribution job=$SLURM_JOB_ID"
