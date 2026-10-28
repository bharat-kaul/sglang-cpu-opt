#!/bin/bash
# LEVER 1 — adaptive-in-M thread count, SINGLE SOCKET of EMR, PHYSICAL CORES ONLY (no SMT).
# Per runtime-config-tuning: fixed-cap OFFLINE sweep on the REAL model, IN-MODEL wall time (isolated
# microbenches mislead), passive wait set FIRST, deterministic rule (not live self-tune), no nested caps.
#
# Control method: restrict the CPU SET itself via numactl --physcpubind to N socket-0 PHYSICAL cores
# (0..N-1; the SMT siblings 120-179 and all of socket 1 are excluded) + --membind=0. This fixes BOTH the
# prefill and decode thread count to N with no SGLang-override battle, and keeps everything single-socket.
# Re-establishes the SINGLE-SOCKET baseline (prior 174.5/57.5 were 2-socket/128-thread) and finds the
# per-phase / per-batch thread optimum = the adaptive-in-M policy.
#
# Sweep: cores N in {8,16,30,45,58} x batch B in {1,32}. One bench_one_batch load gives BOTH prefill and
# decode tok/s for that (N,B). N=58 leaves 2 of 60 cores for the main/framework/OS thread (the bound-2
# headroom rule). A drift re-check re-runs the first leg last.
# Submit: sbatch -p emr --job-name dsv4_thr1s plugin/validate/run_dsv4_threadsweep_1socket.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_thr1s
#SBATCH --output=/scratch/bkaul/dsv4_thr1s_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# wait policy FIRST (set once at launch, before the OMP pool inits) — passive so idle threads park
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 OMP_PROC_BIND=close OMP_PLACES=cores
# single socket, physical cores only: socket 0 phys = 0-59 (120-179 are SMT siblings; socket 1 = 60-119,180-239)
set +x

RES=$WT/plugin/validate/results/dsv4_thr1s_${SLURM_JOB_ID}.csv
echo "cores,batch,prefill_tok_s,decode_tok_s,torch_threads" > "$RES"

run_leg () {  # $1=cores  $2=batch  $3=tag
  local N=$1 B=$2 tag=$3
  echo "===== LEG cores=$N batch=$B (socket0 physical 0-$((N-1))) ====="
  # physcpubind to N socket-0 physical cores; OMP_NUM_THREADS matches; membind socket 0
  OMP_NUM_THREADS=$N numactl --physcpubind=0-$((N-1)) --membind=0 \
    python -m sglang.benchmark.one_batch \
      --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
      --mem-fraction-static 0.80 --batch-size $B --input-len 128 --output-len 16 2>&1 \
    | tee /scratch/bkaul/dsv4_thr1s_${SLURM_JOB_ID}_${tag}.out \
    | grep -iE "Prefill\.|Decode\.|median|torch.*threads|get_num_threads|Error|Traceback|OOM" | tail -8
  # extract and append
  python3 - "$SLURM_JOB_ID" "$tag" "$N" "$B" "$RES" <<'PY'
import sys,re
J,tag,N,B,res=sys.argv[1:6]
try: txt=open(f"/scratch/bkaul/dsv4_thr1s_{J}_{tag}.out").read()
except Exception: txt=""
def grab(pat):
    m=re.findall(pat,txt); return m[-1] if m else ""
pf=grab(r"Prefill\.[^\n]*?throughput:\s*([\d.]+)")
dc=grab(r"Decode\.[^\n]*?throughput:\s*([\d.]+)")
th=grab(r"get_num_threads\(\)=(\d+)") or grab(r"torch.*threads[=:]\s*(\d+)")
open(res,"a").write(f"{N},{B},{pf},{dc},{th}\n")
PY
}

for N in 8 16 30 45 58; do
  run_leg $N 1  "c${N}_b1"
  run_leg $N 32 "c${N}_b32"
done
# drift re-check: repeat the first leg last
run_leg 8 1 "c8_b1_recheck"

echo "############ SINGLE-SOCKET THREAD SWEEP RESULT ############"
column -t -s, "$RES" 2>/dev/null || cat "$RES"
echo "[done] single-socket thread sweep job=$SLURM_JOB_ID"
