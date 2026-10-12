#!/bin/bash
# Parallel re-optimization wave (one kernel per job via KERNEL env). EMR AVX-512, OMP64 pinned, numactl N0.
#SBATCH --job-name=reopt_wave
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/reopt_wave_%x_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_rw_${KERNEL}_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
numactl -N0 -m0 env KERNEL="$KERNEL" python plugin/validate/bench_reopt_wave.py
echo "[done ${KERNEL} ${SLURM_JOB_ID}]"
