#!/bin/bash
# TIME-ATTRIBUTION profile for the pivot chart: per-op wall-time split by PHASE (pf/dec) on real
# DeepSeek-V4-Flash, at realistic serving batch, with the SPIN-WAIT FIX ON (passive) so the split is
# real and not an OpenMP-contention artifact. Short workload (one prefill + a few decode steps).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_pivot
#SBATCH --output=/scratch/bkaul/dsv4_pivot_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
export INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_TIMEIT=1
export INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export INTEL_CPU_DSV4_DECODE_THREADS=8
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size ${BS:-32} --input-len ${ILEN:-512} --output-len ${OLEN:-32} \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.85 --max-total-tokens 32768
echo "[exit=$?]"
echo "=== per-op TIMEIT (pf/dec split) ==="
grep -E "DSV4 TIMEIT|[0-9]+\.[0-9]+s" /scratch/bkaul/dsv4_pivot_${SLURM_JOB_ID}.log || true
