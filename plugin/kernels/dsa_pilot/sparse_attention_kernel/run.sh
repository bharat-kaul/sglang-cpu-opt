#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Launch this script with srun -p emr -t 00:10:00}"
build=$(realpath "${1:?Pass the build directory}")
results=$(realpath -m "${2:?Pass the results directory}")
python=${3:-python3}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$results"
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=threads OMP_WAIT_POLICY=ACTIVE
hostname > "$results/host.txt"
lscpu > "$results/topology.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$results/slurm-job.txt"
sha256sum "$here/sparse_attn.cpp" "$build/bench" "$build/libsparse_attn_emr.so" > "$results/sha256.txt"
for replicate in 1 2 3; do
    extra=()
    if [[ $replicate != 1 ]]; then extra+=(--perf-only); fi
    numactl --physcpubind=0-63 --membind=0 "$python" "$here/bench.py" \
        --build "$build" --output "$results/paired-$replicate.json" \
        --repetitions 35 "${extra[@]}"
    numactl --physcpubind=0-63 --membind=0 "$build/bench" 128 > "$results/native-$replicate.json"
done
numactl --physcpubind=0-63 --membind=0 "$python" "$here/bench.py" \
    --build "$build" --accuracy-only --threads 1 --output "$results/one-thread.json"
numactl --physcpubind=0-63 --membind=0 "$python" "$here/bench.py" \
    --build "$build/ubsan" --accuracy-only --output "$results/ubsan.json"
if [[ ${SPARSE_IMC:-0} == 1 ]]; then
    fifo=$(mktemp -d /tmp/emr-attn.XXXXXX)
    trap 'rm -f "$fifo/control" "$fifo/ack"; rmdir "$fifo"' EXIT
    mkfifo "$fifo/control" "$fifo/ack"
    export SPARSE_PERF_CTL="$fifo/control" SPARSE_PERF_ACK="$fifo/ack"
    numactl --physcpubind=0-63 --membind=0 perf stat -a \
        -e 'uncore_imc_*/cas_count_read/,uncore_imc_*/cas_count_write/' \
        -D -1 --control="fifo:$fifo/control,$fifo/ack" -x, -o "$results/imc.csv" \
        -- "$build/bench" 512 > "$results/imc-native.json"
fi
