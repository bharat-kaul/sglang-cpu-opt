#!/bin/bash
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=olmoe_val
#SBATCH --output=/scratch/bkaul/olmoe_val_%j.log
# OLMoE-1B-7B full validation (BF16 AMX): accuracy parity + gsm8k + throughput.
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
cd /data/nfs_home/bkaul/sglang-cpu-opt
M=/scratch/$USER/models/OLMoE-1B-7B-0924
R=plugin/validate/results
BIND="numactl --cpunodebind=0,1,2 --membind=0,1,2"

echo "[accuracy] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/accuracy_parity.py \
  --model $M --tp 1 --dtype bfloat16 --out $R/olmoe_accuracy.json

echo "[gsm8k] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/task_gsm8k.py \
  --model $M --data /scratch/$USER/gsm8k_test.jsonl --tp 1 \
  --num-questions 100 --num-shots 8 --max-new-tokens 256 --dtype bfloat16 \
  --out $R/olmoe_gsm8k.json

echo "[throughput] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/throughput_bench.py \
  --model $M --config plugin/validate/fixtures/olmoe_config.json --tp 1 \
  --batch 8 --input-len 512 --output-len 64 --ceiling-tflops 60.6 --dtype bfloat16 \
  --out $R/olmoe_throughput.json

echo "[done] $(date +%T)"
