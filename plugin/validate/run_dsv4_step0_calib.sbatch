#!/bin/bash
# DSv4-Flash Phase-2 STEP-0 baseline + depth calibration (kosher baseline: engine as-is, DSA ops torch
# reference, NO optimized kernels wired, canonical pinned config). Runs proxy4 AND proxy8 in ONE job on the
# SAME node+config so t_layer=(t8-t4)/4 and t_fixed=t4-4*t_layer are valid. Operating point: S=1024 input,
# batch sweep {1,8,16,32,64} (B=1 latency floor + B=32 throughput headline), per-step decode (G applied
# analytically). Feeds plugin/validate/results/wall_time_progress.json step 0 (via aggregate/calibration).
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_step0_calib
#SBATCH --output=/scratch/bkaul/dsv4_step0_calib_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1   # route-2 DSA on CPU (torch-reference kernels); disables the GPU paged
                                 # compress-plan nvcc JIT (sgl_kernel_jit_dpsk_v4_compress_plan) that crashes on CPU
export DETERMINISTIC_DUMMY=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO" ]] || { echo "FATAL: cannot resolve a git repository root"; exit 1; }
export PYTHONPATH="$REPO/plugin"
# canonical bind: one socket = one NUMA domain (SNC off), 64 physical cores, pinned ONCE at init
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0   # idle threads must NOT spin (else ~63x prefill / ~10x decode thrash)
cd "$REPO"
INLEN=${INLEN:-1024}; OUTLEN=${OUTLEN:-8}; BATCHES=${BATCHES:-1,8,16,32,64}
echo "node=$(hostname) git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
echo "=== STEP-0 baseline (kosher: torch-reference DSA, no opt kernels wired) S=$INLEN batches=$BATCHES ==="
for TAG in proxy4 proxy8; do
  MODEL=/scratch/$USER/models/flash-v4.1-$TAG
  echo ">>> ===== $TAG ($(python -c "import json;print(json.load(open('$MODEL/config.json'))['num_hidden_layers'])") layers) ====="
  OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 \
    python plugin/validate/bench_wall_time.py \
      --model "$MODEL" --batch-list "$BATCHES" --input-len "$INLEN" --output-len "$OUTLEN" \
      --reps 3 --tp 1 --dtype bfloat16 --load-format dummy --mem-fraction 0.6 \
      --out /scratch/$USER/dsv4_step0_${TAG}_${SLURM_JOB_ID}.json
  echo "[exit=${PIPESTATUS[0]}] $TAG"
done
echo "=== DONE step-0 calibration (proxy4+proxy8) ==="
