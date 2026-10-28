#!/bin/bash
# MILESTONE: repaired, worker-verified MHC A/B at the 8-thread decode operating point, REAL weights,
# single socket, physical cores. Supersedes job 394034 (os.environ-after-Engine toggle = INVALID: both
# arms identical). The repair routes the MHC arm through a WORKER-SIDE control file (re-read per forward,
# crosses the mp.Process boundary) + a provenance counter the driver confirms (reject fallback/no-switch).
# Submit: sbatch -p emr --job-name dsv4_mhc_ab plugin/validate/run_dsv4_mhc_ab_repaired.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_mhc_ab
#SBATCH --output=/scratch/bkaul/dsv4_mhc_ab_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
# 8-thread decode operating point: thread control file re-read per decode forward (valid control pattern).
DTHR=/scratch/bkaul/dsv4_mhc_ab_dthr_${SLURM_JOB_ID}.txt
echo 8 > "$DTHR"
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$DTHR" DECODE_THREADS=8
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0        # park idle threads. NO proc_bind/places (collapse).
set +x
RES=$WT/plugin/validate/results/dsv4_mhc_ab_repaired_${SLURM_JOB_ID}.json
# single socket, physical cores only: 0-57 (SMT 120-179 + socket1 excluded)
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_mhc_ab_repaired.py \
    --model "$CKPT" --config "$CKPT/config.json" --load-format auto \
    --batches 1,32 --input-len 8 --output-len 6 --reps 6 \
    --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[AB\]|\[WARN\]|PARITY|operating_point|gain_pct|Error|Traceback|OOM|assert" | tail -80
echo "############ RESULT ############"; cat "$RES" 2>/dev/null
echo "[done] repaired MHC A/B job=$SLURM_JOB_ID"
