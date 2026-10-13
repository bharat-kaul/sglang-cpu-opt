#!/bin/bash
# Level-2 attribution wave: 4 parallel EMR jobs (one per remaining kernel). Portable (ISA-agnostic) -> EMR farm.
# Array index -> KERNEL. Writes /scratch/$USER/<kernel>_attribution.json (copied back to results/ on collect).
#SBATCH --job-name=kattr
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/%u/kattr_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_kattr_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
KERNELS=(sparse topk sinkhorn combine)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
ATTR_OUT=/scratch/$USER/${K}_attribution.json KERNEL=$K numactl -N0 -m0 python plugin/validate/bench_kernel_attribution.py
echo "[done kattr KERNEL=${K} ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}]"
