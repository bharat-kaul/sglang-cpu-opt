#!/bin/bash
# COUNTERFACTUAL ABLATION: upper-bound the recoverable prefill headroom from fusing/vectorizing the
# attention mask (the measured #1 target: mla.b.mask ~19% of prefill wall). Timing-only; the delta =
# MAX recoverable before authoring a fused-mask attention kernel.
#
# Leg A (baseline) vs Leg B (INTEL_CPU_DSV4_ABLATE_MASK=1, forces the cheap vectorized dense-causal mask
# instead of the per-token Python loop). Prefill tok/s delta = upper bound of the mask-fusion gain.
# RELIABILITY checks (both from the same wall-clock TIMEIT dump):
#   (1) mla.b.mask time must DROP in Leg B  -> the ablation actually took effect (not a no-op flag);
#   (2) moe.expert_apply time must be STABLE A vs B -> rules out a routing-shift confound (wrong
#       attention -> different hidden -> different expert routing -> MoE time changes, which would
#       contaminate the prefill delta).
# REAL 43L weights (real routing), bench_one_batch in-process (TIMEIT atexit dump fires), optimized
# throughput config (spin OMP, all cores, DSA C kernels), B=32 S=128.
# Submit: sbatch -p emr --job-name dsv4_ablmask plugin/validate/run_dsv4_ablate_mask.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_ablmask
#SBATCH --output=/scratch/bkaul/dsv4_ablmask_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
export TIMEIT_MODEL="DeepSeek-V4-Flash" TIMEIT_BATCH=32
set +x

run_leg () {  # $1=tag  $2=ablate(0|1)
  ( local tag=$1 abl=$2
    export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/bkaul/dsv4_ablmask_${SLURM_JOB_ID}_${tag}.json
    if [ "$abl" = "1" ]; then export INTEL_CPU_DSV4_ABLATE_MASK=1; else unset INTEL_CPU_DSV4_ABLATE_MASK; fi
    echo "===== LEG $tag  ABLATE_MASK=${abl} ====="
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
      --mem-fraction-static 0.70 --batch-size 32 --input-len 128 --output-len 4 2>&1 \
      | grep -iE "Prefill|Decode\.|Total\.|token/s|Error|Traceback" | tail -8 )
}
run_leg baseline 0
run_leg ablate   1

echo "############ RECOVERY UPPER BOUND + reliability checks ############"
python3 - "$SLURM_JOB_ID" <<'PY'
import json,sys,glob
J=sys.argv[1]
def leaf(tag,op):
    f=f"/scratch/bkaul/dsv4_ablmask_{J}_{tag}.json"
    try: d=json.load(open(f))
    except Exception as e: return None
    for o in d.get("phases",{}).get("prefill",{}).get("ops",[]):
        if o["op"]==op: return o["s"]
    return 0.0
for tag in ("baseline","ablate"):
    print(f"  {tag:9}: mla.b.mask={leaf(tag,'mla.b.mask')}  moe.expert_apply={leaf(tag,'moe.expert_apply')}  mla.b.pv={leaf(tag,'mla.b.pv')}")
mb=leaf('baseline','mla.b.mask'); ma=leaf('ablate','mla.b.mask')
eb=leaf('baseline','moe.expert_apply'); ea=leaf('ablate','moe.expert_apply')
if mb and ma is not None:
    print(f"\n  (1) mask-time: {mb:.2f}s -> {ma:.2f}s  (ablation took effect if DROPPED)")
if eb and ea:
    drift=abs(ea-eb)/eb*100
    print(f"  (2) moe.expert_apply: {eb:.2f}s -> {ea:.2f}s  ({drift:.0f}% drift; <~10% = clean, no routing confound)")
print("\n  -> compare the printed Prefill tok/s A vs B above; the increase = UPPER BOUND of mask-fusion recovery.")
PY
echo "[done] mask ablation job=$SLURM_JOB_ID"
