#!/bin/bash
# FULL-1319 gsm8k accuracy at the SHIPPED OPTIMIZED config (warm + M-adaptive threads + coop-MHC, shadow build),
# sharded across a slurm job array for fast wall time. Certifies accuracy for the published perf numbers.
# Submit: sbatch -p emr --array=0-15 run_gsm8k_dsv4_sharded_opt.sbatch
# Combine: python plugin/validate/combine_gsm8k_shards.py --glob "/scratch/bkaul/dsv4_gsm8k_shopt_<ARRAYJOBID>_*.json"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_gsm8k_shopt
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_shopt_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# SHIPPED OPTIMIZED config:
export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48" INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000 INTEL_CPU_DSV4_COOP_MHC=1
cd $WT/plugin/validate
python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-1319} --num-shards ${SLURM_ARRAY_TASK_COUNT} --shard-index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size ${CHUNK:-32} --num-shots 8 --max-new-tokens ${MAXNEW:-256} \
    --watchdog-timeout 7000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_shopt_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
    --gens-out /scratch/bkaul/dsv4_gsm8k_shopt_gens_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
echo "[done] shard ${SLURM_ARRAY_TASK_ID}"
