#!/bin/bash
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/bkaul/coop_driver_%j.log
set -x
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
g++ -O3 -march=native -mavx512bf16 -fopenmp $WT/plugin/validate/coop_driver_bench.cpp -o /scratch/bkaul/coop_driver || exit 1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
set +x
for M in 1 32 64; do
  for K in 12 24; do
    echo "########## M=$M K=$K ##########"
    numactl --physcpubind=0-57 --membind=0 /scratch/bkaul/coop_driver 32 $K $M 4096
  done
done
echo "[done] coop driver bench job=$SLURM_JOB_ID"
