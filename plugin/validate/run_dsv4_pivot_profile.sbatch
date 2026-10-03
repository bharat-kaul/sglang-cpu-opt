#!/bin/bash
# TIME-ATTRIBUTION profile for the DSV4 pivot chart: per-op wall-time split by PHASE (pf/dec) on the
# REAL DeepSeek-V4-Flash, SPIN-WAIT FIX ON (passive) so the split is real, not an OpenMP artifact.
# Uses bench_one_batch at tp=1 (runs the forward IN-PROCESS -> the plugin's atexit TIMEIT dump fires;
# the sgl.Engine path hard-kills its worker subprocess and swallows the dump = the old blocker).
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
JSON=/scratch/bkaul/dsv4_pivot_${SLURM_JOB_ID}.json
export INTEL_CPU_DSV4_TIMEIT_JSON=$JSON
export TIMEIT_MODEL="DeepSeek-V4-Flash"
export TIMEIT_NODE="EMR 1-socket (tp=1, decode-cap=8)"
export TIMEIT_PRECISION="MXFP4 W4A16 experts, bf16 compute"
export TIMEIT_BATCH=${BS:-1}
# bench_one_batch at tp=1 runs the forward INLINE in the main process (no worker subprocess), so the
# plugin's atexit TIMEIT dump fires (the sgl.Engine path hard-kills its worker -> dump swallowed = the
# old blocker). One prefill of INLEN tokens + OLEN decode steps; the plugin tags each op pf/dec and
# writes $JSON (pivot schema) on exit, consumed by time_attribution_pivot.py.
python -m sglang.benchmark.one_batch \
  --model-path "$CKPT" --device cpu --tp 1 --trust-remote-code \
  --mem-fraction-static 0.5 \
  --batch-size ${BS:-1} --input-len ${INLEN:-256} --output-len ${OLEN:-16}
echo "[exit=$?] pivot JSON -> $JSON"
if [ -s "$JSON" ]; then echo "JSON WRITTEN"; python3 -c "import json;d=json.load(open('$JSON'));print('phases:',list(d.get('phases',{}).keys()));[print(p,'wall=',d['phases'][p].get('wall_s'),'ops=',len(d['phases'][p]['ops'])) for p in d.get('phases',{})]"; else echo "JSON MISSING (atexit may not have fired)"; fi
