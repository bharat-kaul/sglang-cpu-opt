#!/bin/bash
# GNR DATA-PARALLEL aggregate throughput (Kimi-K3-aligned) — the CORRECT way to measure a many-NUMA node.
# Prior art (runtime-config-tuning SKILL / Kimi-K3 on Xeon): serve a many-NUMA node as ONE NUMA-local rank
# PER DOMAIN (OMP_PROC_BIND=close, OMP_PLACES=cores, OMP_NUM_THREADS=cores/domain, numactl --cpunodebind/
# --membind), set once at launch — NEVER one engine spanning the node (that was run_dsv4_gnr_collect_v2's
# anti-pattern: tp=1 + OMP=256 + interleave=all -> collapsed). Node aggregate throughput = sum of ranks.
#
# Measures, per depth (4L,8L proxy -> two-depth extrapolate to 43L):
#   (A) SINGLE NUMA-local rank (1 domain, no contention)   -> GNR per-domain floor (peer to EMR tp=1 basis)
#   (B) AGGREGATE: one rank per NUMA domain, concurrent     -> GNR full-node throughput (where GNR-AP wins)
# NOTE: prefill is compute-bound (per-domain AMX independent -> aggregate scales cleanly); decode is
# memory-bound and these are DUMMY (small) proxy weights, so the AGG decode sum does NOT capture real
# weight-streaming BW contention across ranks — cross-check AGG decode against the node BW roofline, do
# not publish it as a measured node decode number.
# Submit: sbatch -p gnrap --job-name dsv4_gnr_dp plugin/validate/run_dsv4_gnr_dp_aggregate.sbatch
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_gnr_dp
#SBATCH --output=/scratch/bkaul/dsv4_gnr_dp_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
export PYTHONPATH=$WT/plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
cd "$WT"
set +x

NNODES=$(numactl -H | awk '/^available:/{print $2}')
TPC=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2);print $2}')
N0_LOGICAL=$(numactl -H | awk '/^node 0 cpus:/{print NF-3}')
CPD=$(( N0_LOGICAL / TPC ))   # physical cores per NUMA domain
echo "[topo] node=$NODE numa_domains=$NNODES cores_per_domain=$CPD"
echo "[ref]  EMR single-engine tp=1 (43L): prefill 174.5 / decode 57.5 tok/s (DIFFERENT basis — peer is one GNR domain)."

export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_REAL_ROUTER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DSA_CKERNEL=1
unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS

run_rank () {  # $1=domain $2=layers $3=tag
  local d=$1 NL=$2 tag=$3
  numactl --cpunodebind=$d --membind=$d env \
    OMP_NUM_THREADS=$CPD OMP_PROC_BIND=close OMP_PLACES=cores \
    INTEL_CPU_DSV4_PROV_OUT=$RDIR/dp_${J}_${NODE}_${tag}_d${d}_${NL}L_prov.json \
    python -u $WT/plugin/validate/throughput_bench.py \
      --model "$CKPT" --config "$CKPT/config.json" --tp 1 \
      --batch 32 --input-len 128 --output-len 8 --layers $NL \
      --load-format dummy --mem-fraction 0.10 \
      --out $RDIR/dp_${J}_${NODE}_${tag}_d${d}_${NL}L.json \
      > $RDIR/dp_${J}_${NODE}_${tag}_d${d}_${NL}L.stdout 2>&1
}

for NL in 4 8; do
  echo "===== SINGLE NUMA-local rank d0 ${NL}L (clean per-domain) ====="
  run_rank 0 $NL single
  python3 -c "import json;d=json.load(open('$RDIR/dp_${J}_${NODE}_single_d0_${NL}L.json'));print('  SINGLE d0 %sL: prefill=%.1f decode=%.1f'%('$NL',d['prefill_tok_s'],d['decode_tok_s']))" || tail -5 $RDIR/dp_${J}_${NODE}_single_d0_${NL}L.stdout

  echo "===== AGGREGATE ${NNODES} ranks ${NL}L (one per domain, concurrent) ====="
  pids=()
  for d in $(seq 0 $((NNODES-1))); do run_rank $d $NL agg & pids+=($!); done
  wait "${pids[@]}"
  python3 - "$RDIR" "$J" "$NODE" "$NL" <<'PY'
import json,sys,glob
rdir,J,node,NL=sys.argv[1:5]
pf=dc=0.0; n=0; perrank=[]
for f in sorted(glob.glob(f"{rdir}/dp_{J}_{node}_agg_d*_{NL}L.json")):
    d=json.load(open(f)); pf+=d['prefill_tok_s']; dc+=d['decode_tok_s']; n+=1
    perrank.append((f.split('_d')[-1].split('_')[0], d['prefill_tok_s'], d['decode_tok_s']))
print(f"  AGG {NL}L ranks={n}: prefill_sum={pf:.1f}  decode_sum={dc:.1f} (decode = compute-independent proxy; BW-contention NOT captured)")
for dom,p,c in perrank: print(f"      d{dom}: prefill={p:.1f} decode={c:.1f}")
PY
done
echo "[done] GNR DP aggregate node=$NODE job=$J"
