#!/bin/bash
# EMR PARALLEL ACCURACY WAVE: gsm8k sharded across EMR nodes (correctness is ISA-portable + embarrassingly
# parallel; the harness has --num-shards/--shard-index + a combine). DECODE_THREADS=64 so the batched
# (chunk-16) decode doesn't hit the M=1 thread-starvation. Each shard checkpoints a partial json.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_gsmwave
#SBATCH --output=/scratch/bkaul/dsv4_gsmwave_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=64
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
set +x
python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --num-questions 320 --num-shots 8 --chunk-size 16 --max-new-tokens 256 \
  --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID --mem-fraction 0.70 --acc-min 0.30 \
  --out /scratch/$USER/dsv4_gsmwave_${SLURM_ARRAY_JOB_ID}_sh${SLURM_ARRAY_TASK_ID}.json \
  --gens-out /scratch/$USER/dsv4_gsmwave_${SLURM_ARRAY_JOB_ID}_gens${SLURM_ARRAY_TASK_ID}.json 2>&1 \
  | grep -iE "chunk |acc=|accuracy|verdict|died|OOM|Error|resume"
echo "[exit=${PIPESTATUS[0]}] gsmwave shard $SLURM_ARRAY_TASK_ID"
