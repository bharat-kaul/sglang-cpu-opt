#!/bin/bash
# D2 step-1 REAL-WEIGHT validation + perf: one GNR load proves the all-C/C++ decode chain on REAL
# weights. Four answers:
#  (1) real-weight PARITY: DSA_CKERNEL=1 CHECK=1 asserts every C kernel == eager oracle per call on
#      REAL data (dummy can't check numerics) — the correctness gate.
#  (2) dispatch-provenance: PROV_OUT -> check_dispatch_provenance.py PASS (hot chain all-C/C++).
#  (3) HELD-FIXED decode A/B: ck=0 (eager) vs ck=1 (fused C) on the SAME resident load (CK control
#      file), M=32 threads=8, reps=3 -> the honest perf delta of the wiring (lesson A1).
#  (4) gsm8k accuracy with the C chain vs the 0.797 baseline (no regression).
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_d2_realval
#SBATCH --output=/scratch/bkaul/dsv4_d2_realval_%j.log
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
CTL=/scratch/$USER/dsv4_d2_decthreads_${J}.ctl; echo 8 > "$CTL"
export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL"
CKCTL=/scratch/$USER/dsv4_d2_ckctl_${J}.ctl; echo 1 > "$CKCTL"
export INTEL_CPU_DSV4_CK_CTL_FILE="$CKCTL"

# ---- PASS A: held-fixed eager-vs-C decode A/B (ck sweep) + provenance, no CHECK (clean timing) ----
export INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_PROV_OUT=$RDIR/dsv4_d2_real_prov_${J}.json
export INTEL_CPU_DSV4_SWEEP_JSON=$RDIR/dsv4_d2_real_ckab_${J}.json
set +x
stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --perf-probe --perf-only --perf-m-sweep "32" --perf-thread-sweep "8" --perf-ck-sweep "0,1" \
  --perf-reps 3 --perf-input 128 --perf-output 4 --num-questions 1 --mem-fraction 0.70 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2_realval_${J}_passA.log \
  | stdbuf -oL grep -iE "PERF-SWEEP|died|OOM|Error|Traceback"
echo "[passA exit=${PIPESTATUS[0]}] held-fixed eager-vs-C decode A/B"
python $WT/plugin/validate/check_dispatch_provenance.py $RDIR/dsv4_d2_real_prov_${J}.json; echo "[prov gate exit=$?]"

# ---- PASS B: real-weight PARITY (CHECK=1) + gsm8k accuracy with the C chain ----
set -x
echo 1 > "$CKCTL"
export INTEL_CPU_DSV4_DSA_CKERNEL_CHECK=1
unset INTEL_CPU_DSV4_SWEEP_JSON
set +x
stdbuf -oL python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --num-questions 100 --num-shots 8 --chunk-size 32 --max-new-tokens 256 \
  --mem-fraction 0.70 --acc-min 0.30 \
  --out $RDIR/dsv4_d2_real_gsm8k_${J}.json \
  --gens-out /scratch/$USER/dsv4_d2_real_gens_${J}.json 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_d2_realval_${J}_passB.log \
  | stdbuf -oL grep -iE "chunk |acc=|accuracy|verdict|parity|assert|Error|Traceback|OOM|DSA C-kernel"
echo "[passB exit=${PIPESTATUS[0]}] real-weight parity + gsm8k"
echo "[done] d2 real-weight validation job=$J"
