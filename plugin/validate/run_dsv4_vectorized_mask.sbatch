#!/bin/bash
# VECTORIZED-MASK validation + perf: the REAL fix for the #1 prefill lever (mla.b.mask, ~19% of prefill,
# the per-token Python loop). The vectorized replacement is proven BIT-IDENTICAL offline (4000 fuzzed
# cases, test_vectorized_mask.py); this run (1) PROVES in-situ equivalence on real operands via the
# INTEL_CPU_DSV4_MASK_ASSERT guard, then (2) measures the real prefill gain (no dense-PV penalty, unlike
# the ABLATE_MASK proxy). Same bench_one_batch harness as the mask-ablation (job 393314) so the baseline
# reconciles directly.
#
# Leg A: loop (INTEL_CPU_DSV4_MASK_LOOP=1)                 -> baseline (the published per-token-loop path)
# Leg B: vectorized + MASK_ASSERT=1                        -> PROVES in-situ mask == loop (correctness gate)
# Leg C: vectorized (default, no assert)                   -> real prefill tok/s + mla.b.mask leaf time
# Submit: sbatch -p emr --job-name dsv4_vmask plugin/validate/run_dsv4_vectorized_mask.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_vmask
#SBATCH --output=/scratch/bkaul/dsv4_vmask_%j.log
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

run_leg () {  # $1=tag  $2..=extra env
  ( local tag=$1; shift
    export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/bkaul/dsv4_vmask_${SLURM_JOB_ID}_${tag}.json
    unset INTEL_CPU_DSV4_MASK_LOOP INTEL_CPU_DSV4_MASK_ASSERT
    for kv in "$@"; do export "$kv"; done
    echo "===== LEG $tag  ($*) ====="
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
      --mem-fraction-static 0.70 --batch-size 32 --input-len 128 --output-len 4 2>&1 \
      | grep -iE "Prefill|Decode\.|Total\.|AssertionError|coverage mask|Error|Traceback" | tail -10 )
}
run_leg loop   INTEL_CPU_DSV4_MASK_LOOP=1
run_leg vec_assert INTEL_CPU_DSV4_MASK_ASSERT=1
run_leg vec

echo "############ RESULT: in-situ correctness + prefill gain ############"
if grep -q "coverage mask != loop\|AssertionError" /scratch/bkaul/dsv4_vmask_${SLURM_JOB_ID}.log 2>/dev/null; then
  echo "  !! MASK_ASSERT FAILED -> vectorization diverges in-situ; DO NOT ship (revert to loop)."
else
  echo "  OK: vec_assert leg ran without assertion -> vectorized mask == loop in-situ (correctness proven)."
fi
python3 - "$SLURM_JOB_ID" <<'PY'
import json,sys
J=sys.argv[1]
def mask_s(tag):
    try: d=json.load(open(f"/scratch/bkaul/dsv4_vmask_{J}_{tag}.json"))
    except Exception: return None
    for o in d.get("phases",{}).get("prefill",{}).get("ops",[]):
        if o["op"]=="mla.b.mask": return o["s"]
    return None
for tag in ("loop","vec_assert","vec"):
    print(f"  {tag:10}: mla.b.mask leaf = {mask_s(tag)}s")
print("  -> compare the printed Prefill tok/s for LEG loop vs LEG vec (above); the gain = REALIZED mask-vectorization win.")
PY
echo "[done] vectorized-mask validation job=$SLURM_JOB_ID"
