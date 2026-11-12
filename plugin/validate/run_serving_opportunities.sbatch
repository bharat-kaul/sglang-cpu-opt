#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_serving_screen
#SBATCH --output=/scratch/bkaul/dsv4_serving_screen_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?frozen screening source required}")
MODE="${2:-screen}"
case "$MODE" in screen|meaningful|c4-trial) ;; *) echo "Invalid screen mode: $MODE" >&2; exit 2 ;; esac
OUT="/scratch/bkaul/dsv4-serving-screen/job-${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$OUT/exit_code"
}
trap finish EXIT
test -s "$ROOT/../source-sha256.txt"
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*|ONEDNN_*|DNNL_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*) unset "$key" ;;
    esac
done
export PYTHONPATH="$ROOT/plugin" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1 INTEL_CPU_DSV4_SHARED_CLAMP=1
export INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=0 INTEL_CPU_DSV4_COOP_MHC=0
export INTEL_CPU_DSV4_DECODE_THREADS=32 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=0
export OMP_NUM_THREADS=32 SGLANG_CPU_OMP_THREADS_BIND=0-31
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
export TORCH_EXTENSIONS_DIR="$OUT/extensions" MAX_JOBS=2
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 DISABLE_RADIX=1
hostname > "$OUT/hostname.txt"
lscpu > "$OUT/topology.txt"
numactl --hardware > "$OUT/numa.txt"
cd "$ROOT"
ARGS=()
if [ "$MODE" = meaningful ]; then
    ARGS+=(--meaningful)
elif [ "$MODE" = c4-trial ]; then
    ARGS+=(--c4-trial)
fi
numactl --physcpubind=0-31 --membind=0 \
    /scratch/bkaul/venvs/sglang-cpu/bin/python -u plugin/validate/screen_serving_opportunities.py \
    --output "$OUT/results" "${ARGS[@]}"
if [ "$MODE" = meaningful ]; then
    numactl --physcpubind=0-31 --membind=0 \
        /scratch/bkaul/venvs/sglang-cpu/bin/python -u plugin/validate/replay_serving_sparse.py \
        --results "$OUT/results"
fi
if [ "$MODE" = c4-trial ]; then
    /scratch/bkaul/venvs/sglang-cpu/bin/python -u plugin/validate/analyze_sparse_trial.py \
        --results "$OUT/results"
fi
find "$OUT/extensions" -type f -name '*.so' -exec sha256sum '{}' \; > "$OUT/native-sha256.txt"
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
test -s "$OUT/results/complete.json"
echo "SERVING_SCREEN_JOB_COMPLETE job=$SLURM_JOB_ID"
