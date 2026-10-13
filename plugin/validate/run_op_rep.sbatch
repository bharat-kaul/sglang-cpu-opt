#!/bin/bash
# ONE operating-point replicate (parallel-friendly). REP id from $REP; writes /scratch/$USER/op_rep${REP}.json.
#SBATCH --job-name=op_rep
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/%u/op_rep_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_oprep_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
OP_OUT=/scratch/$USER/op_rep${REP}.json numactl -N0 -m0 python plugin/validate/bench_operating_point.py
echo "[done op_rep REP=${REP} ${SLURM_JOB_ID}]"
