#!/bin/bash
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=olmo2_int8
#SBATCH --output=/scratch/bkaul/olmo2_int8_%j.log
# OLMo-2-7B W8A8-INT8 validation: accuracy parity + gsm8k + throughput.
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
cd /data/nfs_home/bkaul/sglang-cpu-opt
M=/scratch/$USER/models/OLMo-2-1124-7B-w8a8int8
CFG=plugin/validate/fixtures/olmo2_7b_config.json
R=plugin/validate/results
BIND="numactl --cpunodebind=0,1,2 --membind=0,1,2"

echo "[int8 accuracy] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/accuracy_parity.py \
  --model $M --tp 1 --dtype bfloat16 --quantization w8a8_int8 --agree-min 0.75 \
  --out $R/olmo2_7b_int8_accuracy.json

echo "[int8 gsm8k] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/task_gsm8k.py \
  --model $M --data /scratch/$USER/gsm8k_test.jsonl --tp 1 \
  --num-questions 100 --num-shots 8 --max-new-tokens 256 --dtype bfloat16 \
  --quantization w8a8_int8 --out $R/olmo2_7b_int8_gsm8k.json

echo "[int8 throughput] $(date +%T)"
OMP_NUM_THREADS=128 $BIND python plugin/validate/throughput_bench.py \
  --model $M --config $CFG --tp 1 --batch 8 --input-len 512 --output-len 64 \
  --ceiling-tflops 60.6 --dtype bfloat16 --quantization w8a8_int8 \
  --out $R/olmo2_7b_int8_throughput.json

echo "[done] $(date +%T)"
