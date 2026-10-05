#!/bin/bash
# GLM-5.3 Flash gsm8k on the CPU engine — REUSES the DSv4 hard-won harness
# (task_gsm8k_chunked.py: prompt/8-shot/answer-extraction/chunked-checkpoint/sharding),
# adding only GLM's CPU cache config (no_buffer mamba radix + disable_radix + capped context).
#
# QUICK timing check (default):  sbatch run_glm5_gsm8k.sbatch           # 4 Q, chunk 4, 1 EMR node
# FULL sharded run (later):      sbatch --array=0-15 --export=ALL,NQ=1319 run_glm5_gsm8k.sbatch
#   then: python plugin/validate/combine_gsm8k_shards.py --glob "/scratch/bkaul/glm5_gsm8k_sh_<A>_*.json"
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=glm5_gsm8k
#SBATCH --output=/scratch/bkaul/glm5_gsm8k_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
# NUMA single-rank full-node bind (EMR 2x64c) so the 306 GB fp8 model spans the whole node.
export SGLANG_CPU_OMP_THREADS_BIND=${SGLANG_CPU_OMP_THREADS_BIND:-0-127}
# GLM hybrid KDA-mamba + DSA cache config (read by the harness's extra_kw).
export MAMBA_RADIX_STRATEGY=${MAMBA_RADIX_STRATEGY:-no_buffer}
export DISABLE_RADIX=${DISABLE_RADIX:-1}
export CONTEXT_LEN=${CONTEXT_LEN:-8192}
NUMACTL=${NUMACTL:-"numactl --interleave=all"}
AJ=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
AI=${SLURM_ARRAY_TASK_ID:-0}
NS=${SLURM_ARRAY_TASK_COUNT:-1}
# Stagger shard starts (STAGGER secs x shard-index) to avoid NFS thundering-herd on the 306 GB model.
if [[ -n "${STAGGER:-}" && "${AI}" -gt 0 ]]; then echo "STAGGER shard ${AI}: sleep $(( AI * STAGGER ))s"; sleep $(( AI * STAGGER )); fi
cd /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate
$NUMACTL python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/GLM-5.3-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-4} --chunk-size ${CHUNK:-4} \
    --num-shards ${NS} --shard-index ${AI} \
    --num-shots ${SHOTS:-8} --max-new-tokens ${MAXNEW:-256} \
    --mem-fraction ${MEM_FRAC:-0.6} --chunked-prefill-size ${CPS:-512} \
    --watchdog-timeout 7000 \
    --out /scratch/bkaul/glm5_gsm8k_sh_${AJ}_${AI}.json \
    --gens-out /scratch/bkaul/glm5_gsm8k_gens_${AJ}_${AI}.json
echo "[exit=$?]"
