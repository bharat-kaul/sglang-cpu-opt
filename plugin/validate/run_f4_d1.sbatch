#!/bin/bash
# Run the F4 acceptance gate (selftest + manifest incl. the D1 hand-exp compressor) on EMR AVX-512.
#SBATCH --job-name=f4_d1
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/%u/f4_d1_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_f4d1_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
echo ">>> SELFTEST"; numactl -N0 -m0 python plugin/validate/f4_acceptance.py --selftest; echo "selftest_exit=$?"
echo ">>> MANIFEST (incl D1 fp32intrin)"; numactl -N0 -m0 python plugin/validate/f4_acceptance.py; echo "manifest_exit=$?"
