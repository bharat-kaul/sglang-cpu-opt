#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Launch with srun -p emr --constraint=ddr5600 -t 00:10:00}"
build=$(realpath "${1:?Pass build directory}")
results=$(realpath -m "${2:?Pass results directory}")
python=${3:-python3}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$results"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=threads OMP_WAIT_POLICY=ACTIVE
export PYTHONDONTWRITEBYTECODE=1
hostname > "$results/host.txt"
lscpu > "$results/topology.txt"
cp "$build/build.json" "$results/build.json"
cp "$build/fp32_original.cpp" "$results/fp32_original.cpp"
for replicate in 1 2 3; do
    numactl --physcpubind=0-63 --membind=0 "$python" "$here/bench_matched_allocations.py" \
        --build "$build" --replicate "$replicate" --output "$results/replicate-$replicate.json"
done
