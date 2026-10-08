#!/bin/bash
# DSv4-Flash BASELINE wall-time (progressive ledger step 0) — full config, dummy weights, EMR 1 socket.
# E2E decode latency across the M sweep; DSA ENABLED (the ops we optimize live here); one NUMA domain
# bound ONCE (canonical config). Random distinct input tokens -> decorrelated -> MoE routing disperses
# (avoids the broadcast mode-collapse trap). Feeds plugin/validate/results/wall_time_progress.json step 0.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_baseline_wall
#SBATCH --output=/scratch/bkaul/dsv4_baseline_wall_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export DETERMINISTIC_DUMMY=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
# canonical bind: one socket = one NUMA domain (SNC off), 64 physical cores, pinned ONCE at init
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
cd /data/nfs_home/bkaul
MODEL=/scratch/$USER/models/flash-v4.1-dummy
INLEN=2048; OUTLEN=8
echo "=== DSv4-Flash BASELINE wall-time sweep (full 43L/256E dummy, DSA on, 1 socket) ==="
for M in 1 8 16 32 64; do
  echo "########## M=$M ##########"
  OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 \
    python -m sglang.bench_one_batch \
      --model-path "$MODEL" --load-format dummy --device cpu --tp 1 \
      --batch-size "$M" --input-len "$INLEN" --output-len "$OUTLEN" \
      --trust-remote-code --disable-cuda-graph \
      --mem-fraction-static 0.6 --max-total-tokens 180000 2>&1 \
    | grep -iE "Decode|Prefill|latency|throughput|median|OOM|Error" || echo "[M=$M no latency lines]"
  echo "[M=$M exit=${PIPESTATUS[0]}]"
done
echo "=== DONE baseline sweep ==="
