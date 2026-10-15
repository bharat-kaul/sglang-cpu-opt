#!/bin/bash
# Phase-2 integration measurement: E2E wall-time on the PERF-PROXY (4-layer dummy) with the authored DSA C/C++
# kernels OFF (torch baseline) vs ON (INTEL_CPU_DSV4_DSA_CKERNEL=1), at B=1 (decode floor) and B=32 (operating
# point). Uses THIS worktree's plugin (dsv4-tree-pilot) so the C-kernel dispatch is the one under test.
# INTEL_CPU_DSV4_TIMEIT=1 dumps the per-op [DSV4 TIMEIT] breakdown so the delta is attributed to the DSA ops.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_p2integ
#SBATCH --output=/scratch/bkaul/dsv4_p2integ_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_TIMEIT=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_p2integ_${SLURM_JOB_ID}
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
mkdir -p "$TORCH_EXTENSIONS_DIR"
MODEL=/scratch/bkaul/models/DeepSeek-V4-Flash
BENCH=$WT/plugin/validate/throughput_bench.py
LAYERS=${LAYERS:-4}; INLEN=${INLEN:-128}; OUTLEN=${OUTLEN:-32}
cd /data/nfs_home/bkaul
for CK in 0 1; do
  for B in 1 32; do
    DT=$([ "$B" -ge 32 ] && echo 32 || echo 8)
    echo "###### CKERNEL=$CK BATCH=$B decode_threads=$DT ######"
    INTEL_CPU_DSV4_DSA_CKERNEL=$CK INTEL_CPU_DSV4_DECODE_THREADS=$DT python "$BENCH" \
      --model "$MODEL" --config "$MODEL/config.json" --tp 1 \
      --batch "$B" --input-len "$INLEN" --output-len "$OUTLEN" \
      --layers "$LAYERS" --load-format dummy --mem-fraction 0.85 \
      --out /scratch/bkaul/dsv4_p2integ_${SLURM_JOB_ID}_ck${CK}_b${B}.json 2>&1 | tail -40
  done
done
echo "[done p2integ layers=$LAYERS]"
