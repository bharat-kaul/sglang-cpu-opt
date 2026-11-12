#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=dsv4_c4_accuracy
#SBATCH --output=/scratch/bkaul/dsv4_c4_accuracy_%A_%a.log
set -euo pipefail
ROOT=$(realpath "${1:?wave evidence root}")
INDEX=${SLURM_ARRAY_TASK_ID:?six-worker array required}
[[ "$INDEX" =~ ^[0-5]$ ]]
OUT="$ROOT/workers/$INDEX"
test ! -e "$OUT"
mkdir "$OUT"
SLOT=""
CHILD=""
release_slot() {
    if [[ -n "$SLOT" ]]; then
        rm "$SLOT/owner"
        rmdir "$SLOT"
        SLOT=""
    fi
}
finish() {
    rc=$?
    if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
        kill "$CHILD"
        wait "$CHILD" || true
    fi
    release_slot
    printf '%s\n' "$rc" > "$OUT/exit_code"
}
trap finish EXIT
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*|ONEDNN_*|DNNL_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*) unset "$key" ;;
    esac
done
export PYTHONPATH="$ROOT/source/plugin" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1 INTEL_CPU_DSV4_SHARED_CLAMP=1
export INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=0 INTEL_CPU_DSV4_COOP_MHC=0
export INTEL_CPU_DSV4_DECODE_THREADS=32 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=0
export INTEL_CPU_DSV4_CK_STRICT=1
export OMP_NUM_THREADS=32 SGLANG_CPU_OMP_THREADS_BIND=0-31
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
export TORCH_EXTENSIONS_DIR="$OUT/extensions" MAX_JOBS=2
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 DISABLE_RADIX=1
hostname > "$OUT/hostname.txt"
lscpu > "$OUT/topology.txt"
while [[ -z "$SLOT" ]]; do
    for index in 0 1; do
        candidate="$ROOT/load_slots/slot$index"
        if mkdir "$candidate" 2>/dev/null; then
            SLOT="$candidate"
            printf '%s\n' "$SLURM_JOB_ID" > "$SLOT/owner"
            break
        elif [[ ! -d "$candidate" ]]; then
            echo "Cannot create model-load slot $candidate" >&2
            exit 1
        fi
    done
    if [[ -z "$SLOT" ]]; then
        if (( SECONDS > 5400 )); then echo "Model load slot timed out" >&2; exit 1; fi
        sleep 10
    fi
done
echo "C4_MODEL_LOAD_START worker=$INDEX slot=$SLOT time=$(date -Is)"
numactl --physcpubind=0-31 --membind=0 \
    /scratch/bkaul/venvs/sglang-cpu/bin/python -u "$ROOT/source/plugin/validate/c4_accuracy_wave.py" \
    worker --root "$ROOT" --worker "$INDEX" &
CHILD=$!
while [[ ! -s "$OUT/runtime.json" ]]; do
    if ! kill -0 "$CHILD" 2>/dev/null; then
        wait "$CHILD"
        echo "Worker exited before runtime readiness" >&2
        exit 1
    fi
    sleep 5
done
echo "C4_MODEL_LOAD_READY worker=$INDEX time=$(date -Is)"
release_slot
wait "$CHILD"
CHILD=""
find "$OUT/extensions" -type f -name '*.so' -exec sha256sum '{}' \; > "$OUT/native-sha256.txt"
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
test -s "$OUT/complete.json"
echo "C4_ACCURACY_JOB_COMPLETE worker=$INDEX job=$SLURM_JOB_ID"
