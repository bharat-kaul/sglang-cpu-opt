#!/bin/bash
# RUNG-1 COUNTERFACTUAL — decompose the 27% decode inter-op residual to confirm per-layer-fusion recovery,
# BEFORE authoring any fused kernel. Measures decode step latency vs LAYER COUNT (single socket, physical,
# fixed 8 threads, dummy weights + real-router guard, TIMEIT) at N in {4,8,16,32} and fits wall = a + b*N:
#   a (intercept) = PER-STEP FIXED (lm_head/embed/sample) -> NOT recoverable by per-layer fusion
#   b (slope)     = PER-LAYER total -> the per-layer glue fraction (from TIMEIT residual, ~27%) is RECOVERABLE
# => recoverable_by_per_layer_fusion ~= 0.27 * (43*b)/(a + 43*b). Confirms (or shrinks) the 27% claim.
# Dummy weights OK: the dispatch/fork-join/python-loop OVERHEAD is STRUCTURE-driven (transfers to real);
# real-router guard keeps MoE routing realistic. lm_head/model timers (step.lm_head/step.model_total) split
# the per-step-fixed part directly.
# Submit: sbatch -p emr --job-name dsv4_r1cf plugin/validate/run_dsv4_rung1_layerscaling.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_r1cf
#SBATCH --output=/scratch/bkaul/dsv4_r1cf_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_REAL_ROUTER=1       # dummy weights but route like the real model (no mode collapse)
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
set +x
RES=$WT/plugin/validate/results/dsv4_rung1_layerscaling_${SLURM_JOB_ID}.csv
echo "layers,decode_step_ms,layer_total_s,timed_leaf_s,per_layer_glue_s,lm_head_s" > "$RES"

run_leg () {  # $1=N layers
  local N=$1
  local JSON=/scratch/bkaul/dsv4_r1cf_${SLURM_JOB_ID}_L${N}.json
  echo "===== LEG layers=$N ====="
  OMP_NUM_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS=8 \
  INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1 INTEL_CPU_DSV4_TIMEIT_JSON="$JSON" TIMEIT_BATCH=32 \
  numactl --physcpubind=0-57 --membind=0 \
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code --load-format dummy \
      --json-model-override-args "{\"num_hidden_layers\": $N}" \
      --mem-fraction-static 0.70 --batch-size 32 --input-len 128 --output-len 8 2>&1 \
    | tee /scratch/bkaul/dsv4_r1cf_${SLURM_JOB_ID}_L${N}.out \
    | grep -iE "Decode\.|median|DSV4 TIMEIT|step.lm_head|layer.total|Error|Traceback" | tail -6
  python3 - "$SLURM_JOB_ID" "$N" "$JSON" "$RES" <<'PY'
import sys,re,json
J,N,JSON,RES=sys.argv[1:5]
txt=open(f"/scratch/bkaul/dsv4_r1cf_{J}_L{N}.out").read()
dec=re.findall(r"Decode\.[^\n]*?latency:\s*([\d.]+)\s*s[^\n]*?throughput:\s*([\d.]+)",txt)
step_ms=float(dec[-1][0])*1000 if dec else 0.0
try:
    d=json.load(open(JSON)); ph=d["phases"]["decode"]; ops=ph["ops"]; wall=ph.get("wall_s") or 0
    par={"moe","dsa.mla.attend"}
    timed=sum(o["s"] for o in ops if o["op"] not in par)
    glue=max(0.0, wall-timed)
except Exception: wall=timed=glue=0.0
# lm_head from the text dump
lm=re.findall(r"step\.lm_head[^\d]*([\d.]+)\s*s",txt); lm_s=float(lm[-1]) if lm else 0.0
open(RES,"a").write(f"{N},{step_ms:.1f},{wall:.3f},{timed:.3f},{glue:.3f},{lm_s:.3f}\n")
PY
}
for N in 4 8 16 32; do run_leg $N; done

echo "############ RUNG-1 DECOMPOSITION (fit decode_step_ms = a + b*layers) ############"
python3 - "$RES" <<'PY'
import sys,csv
rows=list(csv.DictReader(open(sys.argv[1])))
xs=[int(r["layers"]) for r in rows]; ys=[float(r["decode_step_ms"]) for r in rows]
n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
print(f"fit: decode_step_ms = {a:.0f} (per-step FIXED) + {b:.1f} * layers (PER-LAYER)")
step43=a+b*43
print(f"full 43L step = {step43:.0f} ms = {a:.0f} fixed ({100*a/step43:.0f}%) + {b*43:.0f} per-layer ({100*b*43/step43:.0f}%)")
# per-layer glue fraction from TIMEIT (avg over legs): glue/layer.total
gf=[float(r["per_layer_glue_s"])/float(r["layer_total_s"]) for r in rows if float(r["layer_total_s"])>0]
glue_frac=sum(gf)/len(gf) if gf else 0
print(f"per-layer glue fraction (TIMEIT residual/layer.total) = {100*glue_frac:.0f}% of per-layer time")
rec=glue_frac*(b*43)/step43
print(f"=> recoverable by PER-LAYER FUSION ~= {100*rec:.0f}% of the full decode step "
      f"(per-layer glue {100*glue_frac:.0f}% x per-layer share {100*b*43/step43:.0f}%)")
print(f"   (a full MEGAKERNEL could additionally target the {100*a/step43:.0f}% per-step-fixed, "
      f"dominated by lm_head)")
PY
column -t -s, "$RES" 2>/dev/null || cat "$RES"
echo "[done] rung1 layer-scaling job=$SLURM_JOB_ID"
