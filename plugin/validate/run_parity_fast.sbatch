#!/bin/bash
# FAST accuracy guardrail for the perf-optimization loop: cached-golden next-token parity on
# SHORT prompts (seconds of compute, not the ~50-min gsm8k generation). Run after every kernel
# change to confirm accuracy did not regress. First invocation (no golden yet) CAPTURES the golden
# from the current known-good build; subsequent invocations CHECK against it (exact-match, agree-min 1.0).
#
# Speed levers: (1) golden mode uses short prompts + few tokens => generation is seconds; (2) weights
# are staged to node-local tmpfs (/dev/shm, RAM) so repeated loads on the SAME node skip the NFS read.
# Pin to one node with `sbatch -w <node>` (or --nodelist) so the tmpfs copy is reused across iterations.
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

SRC=/scratch/bkaul/models/DeepSeek-V4-Flash
DST=/dev/shm/$USER/DeepSeek-V4-Flash
# Stage to tmpfs once per node (149GB fits in the ~485GB /dev/shm); NFS read only on first use.
if [ ! -f "$DST/model.safetensors.index.json" ]; then
    mkdir -p "$DST"
    time cp -r "$SRC/." "$DST/"
fi
MODEL="$DST"
GOLDEN=/scratch/bkaul/dsv4_parity_golden.json

cd /data/nfs_home/bkaul
if [ ! -f "$GOLDEN" ]; then
    echo "[parity] capturing golden from current build -> $GOLDEN"
    python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_parity.py \
        --model "$MODEL" --golden-out "$GOLDEN" --max-new 8
else
    echo "[parity] checking current build vs golden $GOLDEN"
    python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_parity.py \
        --model "$MODEL" --golden-in "$GOLDEN" --max-new 8 --agree-min 1.0 \
        --out /scratch/bkaul/dsv4_parity_fast_${SLURM_JOB_ID}.json
fi
echo "[exit=$?]"
