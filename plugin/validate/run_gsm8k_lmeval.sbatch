#!/bin/bash
# STANDARD-HARNESS gsm8k on CPU: run lm-evaluation-harness (the community eval that produces
# published-style numbers) against our sglang CPU DSV4 server. Decomposes the 78-vs-90.8 gap:
# CPU-standard ~= 90.8 -> our custom harness was the gap (CPU correct, gap CLOSED); CPU-standard ~= 78
# -> model genuinely ~78 under a standard harness too. SNEAK-PREVIEW: run small LIMIT first.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4_lmeval
#SBATCH --output=/scratch/bkaul/dsv4_lmeval_%j.log
set -x
PORT=${PORT:-31000}
# 1. sglang CPU server (DSV4 + plugin) in the background
source /scratch/bkaul/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
python -m sglang.launch_server --model-path /scratch/bkaul/models/DeepSeek-V4-Flash \
  --device cpu --tp 1 --trust-remote-code --mem-fraction-static 0.85 \
  --chunked-prefill-size 512 --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!
# 2. wait for the server to come up (model load ~7 min)
READY=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "=== SERVER READY after ${i}0s ==="; READY=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "=== SERVER DIED during load ==="; break; }
  sleep 10
done
[ "$READY" = 1 ] || { echo "server never became ready"; kill $SERVER_PID 2>/dev/null; exit 1; }
# 3. lm-eval (ISOLATED venv) standard gsm8k 8-shot vs the server
/scratch/bkaul/venvs/lmeval/bin/lm_eval \
  --model local-completions \
  --model_args "model=dsv4,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=${CONC:-4},max_retries=2,timeout=${TIMEOUT:-3600},tokenized_requests=False,tokenizer=/scratch/bkaul/models/DeepSeek-V4-Flash" \
  --tasks gsm8k --num_fewshot 8 --limit ${LIMIT:-40} \
  --output_path "/scratch/bkaul/dsv4_lmeval_${SLURM_JOB_ID}" 2>&1 | tail -50
echo "[lm_eval exit=$?]"
kill $SERVER_PID 2>/dev/null
echo "=== gsm8k STANDARD-harness result (vs published 90.8) ==="
grep -riE "exact_match|flexible|strict|gsm8k" "/scratch/bkaul/dsv4_lmeval_${SLURM_JOB_ID}" 2>/dev/null | tail -8
