#!/bin/bash
# Regenerate operating-point data (promoted variants, 512-bit flags, warm+streaming) for the kernel page.
#SBATCH --job-name=op_regen
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/op_regen_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_opregen_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
numactl -N0 -m0 python plugin/validate/bench_operating_point.py
echo "[done op_regen ${SLURM_JOB_ID}]"
