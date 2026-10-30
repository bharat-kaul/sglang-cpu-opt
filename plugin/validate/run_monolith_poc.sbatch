#!/bin/bash
# Monolith thesis counterfactual: per-op fork-join (A) vs one cooperative region/layer (B), warm, 32 threads.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/bkaul/monolith_poc_%j.log
set -x
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
g++ -O3 -march=native -fopenmp $WT/plugin/validate/monolith_thesis_poc.cpp -o /scratch/bkaul/monolith_poc || exit 1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
set +x
for T in 32 48; do
  for OPS in 12 24 40; do
    echo "########## T=$T ops/layer=$OPS ##########"
    numactl --physcpubind=0-57 --membind=0 /scratch/bkaul/monolith_poc $T $OPS 4096
  done
done
echo "[done] monolith poc job=$SLURM_JOB_ID"
