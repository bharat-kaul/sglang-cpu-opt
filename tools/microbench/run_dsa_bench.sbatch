#!/bin/bash
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --job-name=dsa_bench
#SBATCH --output=/scratch/bkaul/dsa_index_gemm_bench_%j.log
set -x
cd /data/nfs_home/bkaul/sglang-cpu-opt/tools/microbench
gcc -O3 -march=native dsa_index_gemm.c -o dsa_index_gemm && echo BUILD_OK
echo "=== LARGE POOL 64MB (memory/gather regime): 1pass then 4pass ==="
for m in 1 0; do numactl --physcpubind=0 --membind=0 ./dsa_index_gemm 16 512 128 262144 2000 $m; done
echo "=== SMALL POOL 256KB (cache-resident regime): 1pass then 4pass ==="
for m in 1 0; do numactl --physcpubind=0 --membind=0 ./dsa_index_gemm 16 512 128 512 4000 $m; done
echo "=== PERF STAT large pool (0=4pass,1=1pass) ==="
for m in 1 0; do echo "-- mode $m --"; perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses numactl --physcpubind=0 --membind=0 ./dsa_index_gemm 16 512 128 262144 2000 $m 2>&1 | grep -iE "mode\"|cycles|instructions|cache-|LLC|insn per cycle" || echo "(perf restricted; timing only above)"; done
echo "[done]"
