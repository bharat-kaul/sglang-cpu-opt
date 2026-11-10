#!/bin/bash
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=dsv4_latest_acc
#SBATCH --output=/scratch/bkaul/dsv4_latest_acc_%A_%a.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
SHARD=${SLURM_ARRAY_TASK_ID:?array task required}
OUT="$ROOT/cpu/shard_$SHARD"
mkdir -p "$OUT"
if [[ -e "$OUT/runtime.json" || -e "$OUT/stamp.txt" || -e "$OUT/score.json" ]]; then
    echo "Refusing to restart a shard that already began model execution: $OUT" >&2
    exit 1
fi
source /scratch/bkaul/venvs/sglang-cpu/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/source/plugin"
export SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1 INTEL_CPU_DSV4_SHARED_CLAMP=1
export INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=0
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 INTEL_CPU_DSV4_COOP_MHC=0
export TORCH_EXTENSIONS_DIR="$OUT/extensions" MAX_JOBS=2
unset DSV4_PREFIX_TRACE_PLAN DSV4_SHARED_REGRESSION_PLAN INTEL_CPU_DSV4_EXECUTION_TRACE
unset INTEL_CPU_DSV4_SHARED_CAPTURE INTEL_CPU_DSV4_DSA_CKERNEL_CHECK
unset INTEL_CPU_DSV4_SHARED_CLAMP_CTL_FILE INTEL_CPU_DSV4_CK_CTL_FILE
SLOT=""
release_slot() {
    if [[ -n "$SLOT" ]]; then
        rm "$SLOT/owner"
        rmdir "$SLOT"
        SLOT=""
    fi
}
finish() {
    RC=$?
    release_slot
    printf '%s\n' "$RC" > "$OUT/exit_code"
}
trap finish EXIT
# Ready engines release their load slot while generation continues independently.
while [[ -z "$SLOT" ]]; do
    LIMIT=2
    if [[ -f "$ROOT/cpu_loader_limit" ]]; then
        read -r LIMIT < "$ROOT/cpu_loader_limit"
    fi
    case "$LIMIT" in 2|4) ;; *) echo "Unsupported loader limit: $LIMIT" >&2; exit 1 ;; esac
    for (( index=0; index<LIMIT; index++ )); do
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
        if (( SECONDS > 5400 )); then echo "Timed out waiting for a model-load slot" >&2; exit 1; fi
        sleep 10
    fi
done
echo "MODEL_LOAD_START shard=$SHARD time=$(date -Is) slot=$SLOT"
bash "$ROOT/source/plugin/validate/run_accuracy_shard.sh" "$ROOT" cpu "$SHARD" 1 &
CHILD=$!
while [[ ! -s "$OUT/runtime.json" ]]; do
    if ! kill -0 "$CHILD" 2>/dev/null; then
        wait "$CHILD"
        echo "Worker exited without a ready runtime snapshot" >&2
        exit 1
    fi
    sleep 5
done
echo "MODEL_LOAD_READY shard=$SHARD time=$(date -Is)"
release_slot
wait "$CHILD"
