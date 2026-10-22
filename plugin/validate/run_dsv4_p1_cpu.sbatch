#!/bin/bash
# P1 CPU paired leg: run the FROZEN token-ID set through CPU-EAGER (ck=0) and CPU-C (ck=1) on ONE
# held-fixed resident load (CK control file) -> the implementation-fidelity equivalence test. CHECK=1
# asserts per-call C-vs-eager parity (safe eager fallback + log if any shape dips); PROV_OUT records
# dispatch. Batched decode (M=N) so it isn't thread-starved.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --job-name=dsv4_p1_cpu
#SBATCH --output=/scratch/bkaul/dsv4_p1_cpu_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=64
# C chain on; CHECK asserts parity (safe fallback if any shape dips); explicit — not inherited.
export INTEL_CPU_DSV4_DSA_CKERNEL=1 INTEL_CPU_DSV4_DSA_CKERNEL_CHECK=1
J=${SLURM_JOB_ID}
CKCTL=/scratch/$USER/dsv4_p1_ck_${J}.ctl; echo 0 > "$CKCTL"
export INTEL_CPU_DSV4_CK_CTL_FILE="$CKCTL"
export INTEL_CPU_DSV4_PROV_OUT=$WT/plugin/validate/results/p1_cpu_prov_${J}.json
cd /data/nfs_home/bkaul
set +x
set -o pipefail
python -u $WT/plugin/validate/paired_eval.py \
  --frozen $WT/plugin/validate/results/p1_frozen_tokenids.json \
  --device cpu --tp 1 --leg cpu --ck-sweep "0,1" --max-new 256 --mem-fraction 0.70 \
  --out-prefix $WT/plugin/validate/results/p1_paired 2>&1 \
  | stdbuf -oL tee /scratch/$USER/dsv4_p1_cpu_${J}_full.log \
  | grep -iE "paired|fell back|parity|Error|Traceback|OOM|died|DSA C-kernel"
RC=${PIPESTATUS[0]}
echo "[exit=${RC}] p1 cpu paired leg"
exit $RC
