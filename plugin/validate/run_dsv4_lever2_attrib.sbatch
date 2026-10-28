#!/bin/bash
# LEVER 2 attribution — split the decode step into kernel-compute vs fork-join/barrier vs inter-op dispatch,
# single socket, physical cores, at the optimum thread count (8, from job 393887). Decides the fusion ROI:
# how much of the ~20x-above-BW-floor decode gap is removable by (per-layer / mega) fusion.
#
# Two legs in ONE job:
#  PART 1 — low-thread extension {1,2,4,8} at B=1,B=32 (resident): if the optimum is 1-2 threads (serial ~=
#           parallel), decode is almost pure fork-join overhead = strong fusion signal.
#  PART 2 — TIMEIT per-op decode attribution at 8 threads, B=32: per-op wall (kernel/torch/framework buckets)
#           + step residual (= inter-op python/dispatch). Σ(op wall) vs step wall vs isolated floors tells us
#           per-op-overhead (grain/fusion) vs inter-op-residual (C++ layer loop / megakernel target).
# Submit: sbatch -p emr --job-name dsv4_l2attr plugin/validate/run_dsv4_lever2_attrib.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --job-name=dsv4_l2attr
#SBATCH --output=/scratch/bkaul/dsv4_l2attr_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0   # NO proc_bind/places (collapse)
set +x

echo "############ PART 1: low-thread extension (fork-join dominance) ############"
CTL=/scratch/bkaul/dsv4_l2attr_ctl_${SLURM_JOB_ID}.txt; echo 8 > "$CTL"
numactl --physcpubind=0-57 --membind=0 env INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL" \
  python -u $WT/plugin/validate/decode_thread_sweep_resident.py \
    --model "$CKPT" --config "$CKPT/config.json" --ctl "$CTL" \
    --threads 1,2,4,8 --batches 1,32 --input-len 8 --output-len 4 --reps 2 \
    --mem-fraction 0.80 --out $WT/plugin/validate/results/dsv4_lowthr_${SLURM_JOB_ID}.json 2>&1 \
  | grep -iE "\[sweep\]|\[OPT\]|Error|Traceback" | tail -20

echo "############ PART 2: per-op decode attribution @ 8 threads, B=32 ############"
JSON=/scratch/bkaul/dsv4_l2attr_pivot_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 env \
  INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1 \
  INTEL_CPU_DSV4_TIMEIT_JSON="$JSON" TIMEIT_BATCH=32 TIMEIT_NODE="EMR 1-socket 8thr" \
  python -m sglang.benchmark.one_batch \
    --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
    --mem-fraction-static 0.80 --batch-size 32 --input-len 128 --output-len 12 2>&1 \
  | grep -iE "Decode\.|median|DECODE THREADCAP|Error|Traceback" | tail -8

echo "############ ATTRIBUTION: per-op decode (kernel/torch/framework) + inter-op residual ############"
python3 - "$JSON" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("no pivot json:",e); sys.exit()
dec=d.get("phases",{}).get("decode",{})
ops=dec.get("ops",[]); wall=dec.get("wall_s") or 0
par={"moe","dsa.mla.attend"}
leaves=[o for o in ops if o["op"] not in par]
bk={}
for o in leaves: bk[o["bucket"]]=bk.get(o["bucket"],0)+o["s"]
timed=sum(bk.values()); resid=max(0.0, wall-timed)
print(f"decode wall(layer.total)={wall:.3f}s  Σ timed-op={timed:.3f}s  inter-op residual(dispatch/python)={resid:.3f}s ({100*resid/wall:.0f}% of wall)")
for b,v in sorted(bk.items(),key=lambda x:-x[1]): print(f"  {b:11} {v:6.3f}s {100*v/wall:5.1f}% of wall")
print("top decode leaves:")
for o in sorted(leaves,key=lambda x:-x['s'])[:10]: print(f"  {o['op']:22} {o['bucket']:10} {o['s']:6.3f}s")
PY
echo "[done] lever2 attribution job=$SLURM_JOB_ID"
