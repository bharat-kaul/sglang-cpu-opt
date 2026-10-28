#!/bin/bash
# Validate fused-layer INCREMENT 1 (MHC sinkhorn+combine C wiring): parity (CHECK) + interleaved A/B.
# Single socket, physical, 8 threads, dummy+real-router, passive. ONE resident load.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_inc1
#SBATCH --output=/scratch/bkaul/dsv4_inc1_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_REAL_ROUTER=1 INTEL_CPU_DSV4_DECODE_THREADS=8
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
set +x
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_mhc_wire_increment1.py \
    --model "$CKPT" --config "$CKPT/config.json" --batch 32 --input-len 8 --output-len 6 --reps 6 \
    --mem-fraction 0.80 --out $WT/plugin/validate/results/dsv4_inc1_mhc_${SLURM_JOB_ID}.json 2>&1 \
  | grep -iE "\[PARITY\]|\[AB\]|MHC|fell back|Error|Traceback|assert" | tail -20
echo "[done] increment1 validation job=$SLURM_JOB_ID"
