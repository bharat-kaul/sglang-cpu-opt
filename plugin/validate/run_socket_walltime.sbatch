#!/bin/bash
# Submit separately to emr/ddr5600 and gnrap; one socket per exclusive node.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_socket_walltime
#SBATCH --output=/scratch/bkaul/dsv4_socket_walltime_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
PLATFORM=${2:?emr or gnr}
[[ "$PLATFORM" == emr || "$PLATFORM" == gnr ]]
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$ROOT/$PLATFORM-exit-code"
}
trap finish EXIT
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*|ONEDNN_*|DNNL_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*) unset "$key" ;;
    esac
done
export PYTHONPATH="$ROOT/source/plugin" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 DISABLE_RADIX=1
/scratch/bkaul/venvs/sglang-cpu/bin/python -u "$ROOT/source/plugin/validate/compare_socket_walltime.py" \
    coordinate --root "$ROOT" --platform "$PLATFORM"
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
test -s "$ROOT/$PLATFORM/complete.json"
echo "SOCKET_WALLTIME_COMPLETE platform=$PLATFORM job=$SLURM_JOB_ID"
