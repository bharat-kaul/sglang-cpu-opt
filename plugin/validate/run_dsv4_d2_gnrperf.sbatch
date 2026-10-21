#!/bin/bash
# D2 GNR PERF + WALL-TIME + BREAKDOWN on a FEW inputs (NOT accuracy — that runs on EMR waves).
# Proves the all-C/C++ decode chain is ACTUALLY used (no silent eager) and measures its perf:
#  (1) STRICT: any kernel fallback RAISES -> completion proves zero silent eager fallbacks.
#  (2) dispatch-provenance PROV_OUT -> check gate: eager==0 across the hot chain (incl MoE).
#  (3) HELD-FIXED eager(0)-vs-C(1) decode A/B on ONE load (CK control file) -> the honest perf delta.
#  (4) TIMEIT component breakdown (per-op wall) + one-decode-step torch.profiler op dump (independent
#      cross-check of the provenance counter: C kernels dominate, no unexpected eager aten op).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_d2_gnrperf
#SBATCH --output=/scratch/bkaul/dsv4_d2_gnrperf_%j.log
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

# ---- PASS A: held-fixed eager-vs-C decode A/B + provenance (STRICT only applies when ck=1 path runs;
#      set STRICT so the ck=1 half proves zero silent fallback; ck=0 half is the eager baseline). ----
export INTEL_CPU_DSV4_DSA_CKERNEL=1 INTEL_CPU_DSV4_CK_STRICT=1
export INTEL_CPU_DSV4_PROV_OUT=$RDIR/dsv4_d2_gnr_prov_${J}.json
export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_d2_gnr_ckab_${J}.json
set +x
stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" --perf-ck-sweep "0,1" \
  --perf-reps 3 --perf-input 512 --perf-output 5 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2_gnrperf_${J}_passA.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|fell back|strict|died|OOM|Error|Traceback"
echo "[passA exit=${PIPESTATUS[0]}] held-fixed eager-vs-C A/B (STRICT)"
python $WT/plugin/validate/check_dispatch_provenance.py $RDIR/dsv4_d2_gnr_prov_${J}.json; echo "[prov gate exit=$?]"

# ---- PASS B: TIMEIT component breakdown + one-decode-step profiler, ck=1 (all-C/C++ chain). ----
set -x
echo 1 > "$CKCTL"
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export INTEL_CPU_DSV4_TIMEIT_JSON=$RDIR/dsv4_d2_gnr_breakdown_${J}.json
export INTEL_CPU_DSV4_PROFILE_OUT=$RDIR/dsv4_d2_gnr_opprofile_${J}.json
unset INTEL_CPU_DSV4_SWEEP_JSON INTEL_CPU_DSV4_PROV_OUT
set +x
stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" \
  --perf-reps 2 --perf-input 512 --perf-output 5 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2_gnrperf_${J}_passB.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|TIMEIT|PROFILE|died|Error|Traceback"
echo "[passB exit=${PIPESTATUS[0]}] component breakdown + op profile"
echo "[done] d2 GNR perf job=$J"
