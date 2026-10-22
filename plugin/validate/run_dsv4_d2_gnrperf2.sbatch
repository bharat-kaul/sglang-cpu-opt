#!/bin/bash
# D2 GNR PERF (right-sized) — the prior run (S=512/G=5/reps=3/2-ck) blew the 2.5h walltime without a
# single row. LIGHT config: S=128, G=3, reps=2. One real-weight load -> three answers:
#  (A) held-fixed eager(ck=0)-vs-C(ck=1) decode A/B, STRICT on (ck=1 completion proves zero silent
#      eager fallback) + dispatch-provenance gate (eager==0 across the hot chain incl MoE).
#  (B) TIMEIT component breakdown (per-op wall) at ck=1.
#  (C) one-decode-step torch.profiler op dump (independent cross-check of the provenance counter).
# NOTE: resident repeated-measurement degrades decode after the first measurement -> reps=2 and small
# G keep each config short; the FIRST rep per config is the trustworthy one (median reported).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_d2_gnrperf2
#SBATCH --output=/scratch/bkaul/dsv4_d2_gnrperf2_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=96
export INTEL_CPU_DSV4_DECODE_THREADS=8
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
cd /data/nfs_home/bkaul
CTL=/scratch/$USER/dsv4_dt_${J}.ctl; echo 8 > "$CTL"; export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL"
CKCTL=/scratch/$USER/dsv4_ck_${J}.ctl; echo 1 > "$CKCTL"; export INTEL_CPU_DSV4_CK_CTL_FILE="$CKCTL"

# ---- held-fixed eager-vs-C A/B + provenance + breakdown + profiler, ONE load (perf-only). ----
export INTEL_CPU_DSV4_DSA_CKERNEL=1 INTEL_CPU_DSV4_CK_STRICT=1
export INTEL_CPU_DSV4_PROV_OUT=$RDIR/dsv4_d2_gnr2_prov_${J}.json
export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_d2_gnr2_ckab_${J}.json
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export INTEL_CPU_DSV4_TIMEIT_JSON=$RDIR/dsv4_d2_gnr2_breakdown_${J}.json
export INTEL_CPU_DSV4_PROFILE_OUT=$RDIR/dsv4_d2_gnr2_opprofile_${J}.json
set +x
stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" --perf-ck-sweep "0,1" \
  --perf-reps 2 --perf-input 128 --perf-output 3 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2_gnrperf2_${J}_full.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|TIMEIT|PROFILE|fell back|strict|died|OOM|Error|Traceback"
echo "[exit=${PIPESTATUS[0]}] perf A/B + breakdown"
python $WT/plugin/validate/check_dispatch_provenance.py $RDIR/dsv4_d2_gnr2_prov_${J}.json; echo "[prov gate exit=$?]"
echo "[done] d2 GNR perf2 job=$J"
