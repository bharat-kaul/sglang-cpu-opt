#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_c4_crossover
#SBATCH --output=/scratch/bkaul/dsv4_c4_crossover_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?frozen root}")
OUT="$ROOT/job-$SLURM_JOB_ID"
mkdir "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"' EXIT
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export PYTHONPATH="$ROOT/source/plugin" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
hostname > "$OUT/hostname.txt"
lscpu > "$OUT/topology.txt"
cd "$ROOT/source"
for replicate in 0 1 2; do
    numactl --physcpubind=0-31 --membind=0 \
        /scratch/bkaul/venvs/sglang-cpu/bin/python -u plugin/validate/screen_c4_crossover.py \
        --output "$OUT/replicate$replicate" --replicate "$replicate"
done
(cd "$ROOT/source" && sha256sum --quiet -c ../source-sha256.txt)
echo "C4_CROSSOVER_COMPLETE job=$SLURM_JOB_ID"
