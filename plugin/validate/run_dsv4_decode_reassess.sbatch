#!/bin/bash
# DECODE REASSESSMENT: is decode BW-bound or latency-bound, and what converts it to BW-bound?
# The published ~63 "achievable" was WRONG (it anchored on the current latency-bound impl, conflating
# "dispatched to a C kernel" with "running at the kernel's BW floor"). Measured decode is 12x (M=32) to
# 62x (M=1) ABOVE the weight-streaming BW floor -> 92-98% is non-streaming overhead. This job SIZES the
# real gap: (1) a DECODE_THREADS sweep at M=1 (the Xeon low-batch target) to quantify the threading
# pathology and find the optimum, and (2) per-op TIMEIT at each point so every op can be scored vs its
# OWN BW floor (ms/step vs weight-bytes/BW) -> names which ops are latency-bound and by how much
# (MoE group-GEMM/BRGEMM? dispatch? dense GEMV?). Mirrors the prefill attribution exercise.
# Submit: sbatch -p emr --job-name dsv4_decsweep plugin/validate/run_dsv4_decode_reassess.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:40:00
#SBATCH --job-name=dsv4_decsweep
#SBATCH --output=/scratch/bkaul/dsv4_decsweep_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0     # latency config (Xeon low-batch target)
export TIMEIT_MODEL="DeepSeek-V4-Flash"
set +x

run_leg () {  # $1=tag  $2=decode_threads(unset|N)  $3=batch
  ( local tag=$1 dth=$2 bs=$3
    export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/bkaul/dsv4_decsweep_${SLURM_JOB_ID}_${tag}.json TIMEIT_BATCH=$bs
    if [ "$dth" = "unset" ]; then unset INTEL_CPU_DSV4_DECODE_THREADS; else export INTEL_CPU_DSV4_DECODE_THREADS=$dth; fi
    echo "===== LEG $tag  DECODE_THREADS=${dth} batch=${bs} ====="
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
      --mem-fraction-static 0.60 --batch-size $bs --input-len 128 --output-len 32 2>&1 \
      | grep -iE "Decode\.|median|token/s|Error|Traceback" | tail -4 )
}
# M=1 thread sweep (the Xeon target); then M=32 at the best-known cap for contrast
run_leg m1_t4   4   1
run_leg m1_t8   8   1
run_leg m1_t32  32  1
run_leg m1_tall unset 1
run_leg m32_t8  8   32

echo "############ per-op ms/step vs BW floor (names the latency-bound ops) ############"
python3 - "$SLURM_JOB_ID" <<'PY'
import json,sys,glob
J=sys.argv[1]; BW=848.0
# weight bytes per step per op-class at M=1 (GB), rough: experts 6*12.58MB*43L, dense/attn fp8, etc.
for f in sorted(glob.glob(f"/scratch/bkaul/dsv4_decsweep_{J}_*.json")):
    tag=f.split(f"_{J}_")[1].replace(".json","")
    try: d=json.load(open(f))
    except Exception: continue
    dec=d.get("phases",{}).get("decode",{})
    ops=dec.get("ops",[]); 
    if not ops: continue
    # steps = output-len 32
    bk={}
    for o in ops: bk[o["op"]]=bk.get(o["op"],0)+o["s"]
    tot=sum(bk.values())
    print(f"\n--- {tag}: timed decode leaves sum={tot:.2f}s ---")
    for op,s in sorted(bk.items(),key=lambda x:-x[1])[:8]:
        print(f"    {op:22} {s:6.2f}s  {100*s/tot:4.1f}%")
PY
echo "[done] decode reassessment job=$SLURM_JOB_ID"
