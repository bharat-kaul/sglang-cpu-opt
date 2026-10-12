#!/bin/bash
# Compressor vectorization retry (RAJAPerf hpc-kernel-optimization SKILL): build-flag matrix +
# asm classification + variant comparison. EMR to match the published compressor baseline (job 385001).
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=cz_vectorize
#SBATCH --output=/scratch/bkaul/cz_vectorize_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_cz_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot/plugin/validate
numactl -N0 -m0 python bench_compressor_vectorize.py
echo "[done] $SLURM_JOB_ID"
