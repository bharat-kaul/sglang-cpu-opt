#!/bin/bash
# FAST accuracy guardrail for the perf-optimization loop: cached-golden next-token parity on
# SHORT prompts (seconds of compute, not the ~50-min gsm8k generation). Run after every kernel
# change to confirm accuracy did not regress. First invocation (no golden yet) CAPTURES the golden
# from the current known-good build; subsequent invocations CHECK against it (exact-match, agree-min 1.0).
#
# NOTE: do NOT stage this 149GB model to /dev/shm (tmpfs is RAM-backed) — it competes with the model's
# own loaded/dequantized footprint + KV reservation and OOM-kills the forward. Load from NFS directly
# (~8 min). The real speedup here is SHORT prompts (seconds of gen), not load. lower MEM_FRAC to be safe.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_parity_fast
#SBATCH --output=/scratch/bkaul/dsv4_parity_fast_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export MEM_FRAC=0.5
export WATCHDOG=3600

MODEL=/scratch/bkaul/models/DeepSeek-V4-Flash
GOLDEN=/scratch/bkaul/dsv4_parity_golden.json

cd /data/nfs_home/bkaul
# decode is slow pre-optimization; keep max-new small (next-token parity is the key regression signal).
if [ ! -f "$GOLDEN" ]; then
    echo "[parity] capturing golden from current build -> $GOLDEN"
    python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_parity.py \
        --model "$MODEL" --golden-out "$GOLDEN" --max-new 2
else
    echo "[parity] checking current build vs golden $GOLDEN"
    python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_parity.py \
        --model "$MODEL" --golden-in "$GOLDEN" --max-new 2 --agree-min 1.0 \
        --out /scratch/bkaul/dsv4_parity_fast_${SLURM_JOB_ID}.json
fi
echo "[exit=$?]"
