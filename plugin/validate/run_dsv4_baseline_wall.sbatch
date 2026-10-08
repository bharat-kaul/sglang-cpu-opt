#!/bin/bash
# DSv4-Flash BASELINE wall-time (progressive ledger step 0) — dummy weights, EMR 1 socket.
# Perf runs use DUMMY weights (no real load) AND a depth-reduced proxy: every DSv4 layer is the
# identical MoE+MLA+DSA archetype (first_k_dense_replace=0, moe_layer_freq=1), so a 4-layer proxy
# (flash-v4.1-proxy4, full width/ranks/experts/groups) gives representative per-op timing and
# progression deltas without paying the 43x compute or the 139GB dummy-init. Override MODEL for a
# one-off full-depth anchor (flash-v4.1-dummy). DSA ENABLED (the ops we optimize live here); one
# NUMA domain bound ONCE. Random distinct input tokens -> decorrelated -> MoE routing disperses
# (avoids the broadcast mode-collapse trap). Feeds plugin/validate/results/wall_time_progress.json.
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
MODEL=${MODEL:-/scratch/$USER/models/flash-v4.1-proxy4}
INLEN=2048; OUTLEN=8
TAG=$(basename "$MODEL")
echo "=== DSv4-Flash BASELINE wall-time  model=$MODEL (DSA on, 1 socket, Engine path) ==="
OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 \
  python plugin/validate/bench_wall_time.py \
    --model "$MODEL" --batch-list 1,8,16,32,64 --input-len "$INLEN" --output-len "$OUTLEN" \
    --reps 3 --tp 1 --dtype bfloat16 --load-format dummy --mem-fraction 0.6 \
    --out /scratch/$USER/dsv4_wall_${TAG}.json
echo "[exit=$?]"
echo "=== DONE baseline sweep ($TAG) ==="
