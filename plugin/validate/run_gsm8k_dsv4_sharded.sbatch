#!/bin/bash
# PARALLEL gsm8k for DeepSeek-V4-Flash: a slurm JOB ARRAY shards the (independent)
# questions across N EMR nodes; each shard is chunked+checkpointed; combine sums them.
# gsm8k questions are independent -> embarrassingly parallel -> ~N x faster wall-clock,
# and (NFS caveat) N nodes each read the 149 GB model, so keep N moderate.
# Submit:  sbatch run_gsm8k_dsv4_sharded.sbatch            # 8 shards, full 1319 Q
# Combine: python combine_gsm8k_shards.py --glob "/scratch/bkaul/dsv4_gsm8k_sh_<ARRAYJOBID>_*.json"
#SBATCH --partition=emr
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_gsm8k_sh
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_sh_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
cd /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate
# Cold-cache mitigation (default OFF): stagger shard starts so the 149 GB loads pipeline
# through NFS instead of a thundering herd. MEASURED unnecessary when the NFS server cache
# is warm (8x concurrent load ~2.5 min/node < 7 min cold single). Set STAGGER=300 (s) only
# if the shard logs show a COLD/contended load.
sleep $(( SLURM_ARRAY_TASK_ID * ${STAGGER:-0} ))
python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-1319} \
    --num-shards ${SLURM_ARRAY_TASK_COUNT} --shard-index ${SLURM_ARRAY_TASK_ID} \
    --chunk-size ${CHUNK:-32} --num-shots 8 --max-new-tokens ${MAXNEW:-256} \
    --watchdog-timeout 7000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_sh_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
echo "[exit=$?]"
