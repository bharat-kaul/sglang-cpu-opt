#!/bin/bash
# Feasibility microbench: is B=32's ~100 GB/s the scattered-expert access-pattern ceiling, or recoverable
# memory-level parallelism? Compiles + runs moe_bw_feasibility.cpp on one EMR socket at T=32/48 for the
# B=32 (56 experts) and B=64 (203 experts) access patterns. Submit:
#   sbatch -p emr --job-name moe_bw_feas plugin/validate/run_moe_bw_feasibility.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/bkaul/moe_bw_feas_%j.log
set -x
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
g++ -O3 -march=native -mavx512bw -fopenmp $WT/plugin/validate/moe_bw_feasibility.cpp -o /scratch/bkaul/moe_bw_feas || exit 1
export OMP_PROC_BIND=close OMP_PLACES=cores
set +x
for T in 32 48; do
  for A in 56 203; do
    echo "########## T=$T active=$A ##########"
    numactl --physcpubind=0-57 --membind=0 /scratch/bkaul/moe_bw_feas $T $A
  done
done
echo "[done] moe bw feasibility job=$SLURM_JOB_ID"
