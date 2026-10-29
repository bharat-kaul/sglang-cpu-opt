#!/bin/bash
# RE-ATTRIBUTION at the WARM operating point: size the monolith's residual prize after the warm-pool config
# win. Same TIMEIT per-op decode attribution at the 8-thread STICKY point, run twice (cold vs warm, fresh
# one_batch each), identical but for OMP_WAIT_POLICY/KMP_BLOCKTIME. Residual = layer.total - Σ timed ops =
# inter-op dispatch/python/copies. Cold residual should reproduce ~29%; warm residual = what fusion could
# still recover (dispatch+copies after park/wake is gone).
# Submit: sbatch -p emr --job-name dsv4_attr_wc plugin/validate/run_dsv4_attrib_warmcold.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_attr_wc
#SBATCH --output=/scratch/bkaul/dsv4_attr_wc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# 8-thread sticky decode cap + TIMEIT per-op, in both arms
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
set +x

run_attr () {  # $1=tag  $2=wait_policy  $3=blocktime  $4=spincount
  local TAG=$1 JSON=/scratch/bkaul/dsv4_attr_${1}_${SLURM_JOB_ID}.json
  echo "############ ATTRIBUTION ARM: $TAG (OMP_WAIT_POLICY=$2 KMP_BLOCKTIME=$3) ############"
  numactl --physcpubind=0-57 --membind=0 env \
    OMP_WAIT_POLICY=$2 KMP_BLOCKTIME=$3 GOMP_SPINCOUNT=$4 \
    INTEL_CPU_DSV4_TIMEIT_JSON="$JSON" TIMEIT_BATCH=32 TIMEIT_NODE="EMR 1sock 8thr $TAG" \
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
      --mem-fraction-static 0.80 --batch-size 32 --input-len 8 --output-len 12 2>&1 \
    | grep -iE "Decode\.|median|DECODE THREADCAP|Error|Traceback" | tail -6
  python3 - "$JSON" "$TAG" <<'PY'
import json,sys
tag=sys.argv[2]
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(f"[{tag}] no pivot json:",e); sys.exit()
dec=d.get("phases",{}).get("decode",{})
ops=dec.get("ops",[]); wall=dec.get("wall_s") or 0
par={"moe","dsa.mla.attend"}
leaves=[o for o in ops if o["op"] not in par]
bk={}
for o in leaves: bk[o["bucket"]]=bk.get(o["bucket"],0)+o["s"]
timed=sum(bk.values()); resid=max(0.0, wall-timed)
print(f"[{tag}] decode wall(layer.total)={wall:.3f}s  Σtimed={timed:.3f}s  inter-op residual={resid:.3f}s ({100*resid/wall:.0f}% of wall)")
for b,v in sorted(bk.items(),key=lambda x:-x[1]): print(f"  [{tag}] {b:11} {v:6.3f}s {100*v/wall:5.1f}%")
for o in sorted(leaves,key=lambda x:-x['s'])[:8]: print(f"  [{tag}] leaf {o['op']:22} {o['bucket']:10} {o['s']:6.3f}s")
PY
}

run_attr cold passive 0 0
run_attr warm active infinite infinite
echo "[done] warm-vs-cold attribution job=$SLURM_JOB_ID"
