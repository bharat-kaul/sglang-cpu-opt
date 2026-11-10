#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=q1175_cpu_trace
#SBATCH --output=/scratch/bkaul/q1175_cpu_trace_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
SOURCE="$ROOT/source/plugin"
source /scratch/bkaul/venvs/sglang-cpu/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE"
export SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1
export INTEL_CPU_DSV4_SHARED_CLAMP=1 INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=0
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 INTEL_CPU_DSV4_COOP_MHC=0
export TORCH_EXTENSIONS_DIR="$ROOT/extensions_$SLURM_JOB_ID" MAX_JOBS=2
python -u "$SOURCE/validate/run_prefix_trace.py" "$ROOT/plan.json" "$ROOT/cpu_$SLURM_JOB_ID" --device cpu --batch-index "${2:?batch index}"
