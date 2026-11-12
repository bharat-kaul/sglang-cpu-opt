#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --job-name=dsv4_comp_batch
#SBATCH --output=/scratch/bkaul/dsv4_comp_batch_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?frozen source required}")
OUT="/scratch/bkaul/dsv4-compression-batch/job-${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"' EXIT
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*) unset "$key" ;;
    esac
done
export PYTHONPATH="$ROOT/plugin" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_CPU_ENGINE=1 OMP_NUM_THREADS=32
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
hostname > "$OUT/hostname.txt"
lscpu > "$OUT/topology.txt"
cd "$ROOT"
numactl --physcpubind=0-31 --membind=0 \
    /scratch/bkaul/venvs/sglang-cpu/bin/python -u plugin/validate/bench_compression_batch.py \
    --output "$OUT/results"
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
test -s "$OUT/results/complete.json"
echo "COMPRESSION_BATCH_JOB_COMPLETE job=$SLURM_JOB_ID"
