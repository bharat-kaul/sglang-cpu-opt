#!/bin/bash
# ACCURACY PARITY: gsm8k exact-match under the warm config vs the cold-ref baseline on the SAME subset.
# differ ONLY by threading env (DTHR/WAITP/SPIN). Confirms the ~4-5x warm config preserves accuracy.
# for ref:  sbatch --export=ALL,TAG=ref,DTHR=8,WAITP=passive,SPIN=0,NQ=200 ...
# for warm: sbatch --export=ALL,TAG=warm,DTHR=32,WAITP=active,SPIN=infinite,NQ=200 ...
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_wc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=${DTHR:-8}
export OMP_WAIT_POLICY=${WAITP:-passive} KMP_BLOCKTIME=0 GOMP_SPINCOUNT=${SPIN:-0}
set +x
RES=$WT/plugin/validate/results/dsv4_gsm8k_${TAG:-x}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python $WT/plugin/validate/task_gsm8k.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-200} --num-shots 8 --max-new-tokens ${MAXNEW:-256} \
    --watchdog-timeout 8400 --chunked-prefill-size 512 --out "$RES" 2>&1 \
  | grep -iE "accuracy|exact|correct|questions|Error|Traceback|OOM" | tail -15
echo "########## RESULT ($TAG DTHR=$DTHR WAITP=$WAITP SPIN=$SPIN) ##########"; cat "$RES" 2>/dev/null
echo "[done gsm8k warmcheck $TAG job=$SLURM_JOB_ID]"
