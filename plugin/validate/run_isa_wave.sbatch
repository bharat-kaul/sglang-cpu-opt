#!/bin/bash
# ISA + FUSION wave: completes the indexer conformance-safe (bf16-KV) / bit-exact (dpbf16) levers and probes the
# sparse own-fused path + the dispatch-bound ISA knob. 4 parallel EMR tasks = {sparse,indexer} x {default AMX,
# forced AVX-512-bf16 (dpbf16, AMX disabled via ONEDNN_MAX_CPU_ISA)}. The forced-ISA run makes the bf16 at::bmm
# variants use AVX-512-bf16 (dpbf16) instead of AMX -> the conformance-safe low-M ISA test, donor-first (no hand
# micro-kernel). Writes /scratch/$USER/reference_wave_<kernel>_<isa>.json.
#SBATCH --job-name=isawave
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/%u/isawave_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_isawave_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd /data/nfs_home/bkaul/dsv4-tree-pilot
KERNELS=(sparse sparse indexer indexer)
ISAS=(default avx512bf16 default avx512bf16)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
ISA=${ISAS[$SLURM_ARRAY_TASK_ID]}
if [ "$ISA" = "avx512bf16" ]; then
  export ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 DNNL_MAX_CPU_ISA=AVX512_CORE_BF16   # disable AMX -> dpbf16 path
fi
WAVE_OUT=/scratch/$USER/reference_wave_${K}_${ISA}.json numactl -N0 -m0 python plugin/validate/bench_reference_wave.py --kernel $K
echo "[done isawave KERNEL=${K} ISA=${ISA} ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}]"
