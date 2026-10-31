#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_gsm8k_shbase
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_shbase_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
# BASELINE config: cold wait, 8-thread decode cap, NO coop, INSTALLED build (no shadow)
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 INTEL_CPU_DSV4_COOP_MHC=0
cd $WT/plugin/validate
python task_gsm8k_chunked.py --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions 1319 --num-shards ${SLURM_ARRAY_TASK_COUNT} --shard-index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 32 --num-shots 8 --max-new-tokens 256 --watchdog-timeout 7000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_shbase_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
echo "[done] base shard ${SLURM_ARRAY_TASK_ID}"
