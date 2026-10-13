#!/bin/bash
# Level-2 RECOVERY wave: 3 parallel EMR jobs (sparse/topk/sinkhorn) testing the attribution-motivated knob.
# combine excluded (at-ceiling confirmed: op_read ~= raw-read BW). Writes /scratch/$USER/recovery_<kernel>.json.
#SBATCH --job-name=krec
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/%u/krec_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_krec_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
KERNELS=(sparse topk sinkhorn)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
REC_OUT=/scratch/$USER/recovery_${K}.json KERNEL=$K numactl -N0 -m0 python plugin/validate/bench_kernel_recovery.py
echo "[done krec KERNEL=${K} ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}]"
