#!/bin/bash
# MoE CAMPAIGN wave-1: warm/cold x thread-count x B-sweep, worker-accurate phase-separated timing + MoE
# occupancy/branch probe. Parameterized via --export:
#   WARM=0|1         cold(passive/blocktime0) vs warm(active/blocktime inf)
#   DTHR=<n>         decode thread cap
#   STICKY=0|1       1=sticky (clean decode, prefill@DTHR) ; 0=restore (prefill@58, realistic churn)
# Example matrix:
#   # decode thread screen (sticky): does warm decode want >8?
#   for T in 8 16 32 58; do sbatch -p emr --export=ALL,WARM=1,DTHR=$T,STICKY=1 --job-name moec_w$T \
#       plugin/validate/run_dsv4_moe_campaign.sbatch; done
#   sbatch -p emr --export=ALL,WARM=0,DTHR=8,STICKY=1 --job-name moec_c8 plugin/validate/run_dsv4_moe_campaign.sbatch
#   # deploy mode (non-sticky): prefill@58 + decode@8, true prefill regression
#   for W in 0 1; do sbatch -p emr --export=ALL,WARM=$W,DTHR=8,STICKY=0 --job-name moec_dep$W \
#       plugin/validate/run_dsv4_moe_campaign.sbatch; done
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/bkaul/dsv4_moec_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
WARM=${WARM:-1}; DTHR=${DTHR:-8}; STICKY=${STICKY:-1}
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=$DTHR INTEL_CPU_DSV4_DECODE_THREADS_STICKY=$STICKY
export INTEL_CPU_DSV4_MOE_OCC=1 INTEL_CPU_DSV4_MOE_OCC_FILE=$WT/plugin/validate/results/moec_occ_${SLURM_JOB_ID}.json
if [ "$WARM" = "1" ]; then export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite GOMP_SPINCOUNT=${SPIN:-infinite}; TAG=warm${SPIN:+_sc$SPIN}; else export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0; TAG=cold; fi
export CLEAN_CORES="${TAG}/DTHR${DTHR}/sticky${STICKY}"
set +x
RES=$WT/plugin/validate/results/moec_${TAG}_T${DTHR}_s${STICKY}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" \
    --batches ${BATCHES:-1,2,4,8,16,32,64} --input-len 8 --output-len ${OUTLEN:-8} --reps ${REPS:-3} --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[CLEAN\]|\[DECODE THREADCAP\]|decode_step|Error|Traceback|OOM" | tail -40
echo "############ RESULT ($CLEAN_CORES) ############"; cat "$RES" 2>/dev/null
echo "############ OCC ############"; cat $WT/plugin/validate/results/moec_occ_${SLURM_JOB_ID}.json 2>/dev/null
echo "[done] moe campaign $CLEAN_CORES job=$SLURM_JOB_ID"
