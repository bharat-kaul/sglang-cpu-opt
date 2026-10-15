#!/bin/bash
# REFERENCE-ALGORITHM wave: 2 parallel EMR jobs (sparse/indexer) measuring the kernels at the dtype the
# DeepSeek reference (inference/kernel.py) uses (bf16-native, no fp32->bf16 cast) vs the CORRECTED roofline,
# with cos vs the fp32 oracle. Isolates the cast confound that sank the earlier bf16 attempts. One node per
# kernel. Writes /scratch/$USER/reference_wave_<kernel>.json.
#SBATCH --job-name=refwave
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/%u/refwave_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_refwave_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
KERNELS=(sparse indexer)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
WAVE_OUT=/scratch/$USER/reference_wave_${K}.json numactl -N0 -m0 python plugin/validate/bench_reference_wave.py --kernel $K
echo "[done refwave KERNEL=${K} ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}]"
