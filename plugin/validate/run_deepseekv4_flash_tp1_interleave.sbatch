#!/bin/bash
# DeepSeek-V4-Flash decode roofline at tp=1 with INTERLEAVED memory across the whole node.
# Rationale: tp>1 on CPU pays SGLang's shared-memory TP AllReduce, which is ~750x slower than
# even gloo in-model (a pathology, ~6s/layer) and makes tp=4 decode impractical. tp=1 has NO
# AllReduce and the decode thread cap works. Flash fp8 (~292GB after fp4->fp8) exceeds one
# 248GB NUMA domain, so interleave it across the node's ~1.5TB: numactl --interleave=all for
# physical placement + INTEL_CPU_DSV4_INTERLEAVE_MEM=1 so SGLang reports full-node memory.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_flash_tp1i
#SBATCH --output=/scratch/bkaul/dsv4_flash_tp1i_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export INTEL_CPU_DSV4_DSA_CPU=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
unset INTEL_CPU_DSV4_BYPASS_DSA
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export SGLANG_AUTO_NUMA_BIND=0                # don't membind rank to node0 (fights interleave)
export SGLANG_NUMA_BIND_V2=0
export INTEL_CPU_DSV4_DECODE_THREADS=auto     # tp=1 -> per-forward cap works (200x M=1 fix)
export INTEL_CPU_DSV4_INTERLEAVE_MEM=1        # report full-node mem + re-apply interleave policy
export INTEL_CPU_DSV4_TIMEIT=1
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
# Background per-node RSS sampler: verify the interleave policy actually spreads pages
# (skill: "verify placement, don't assume it"). Logs numastat every 30s for the python PID.
( for _ in $(seq 1 40); do sleep 30; p=$(pgrep -f 'bench_one_batch' | head -1); \
    [ -n "$p" ] && echo "[numastat $(date +%H:%M:%S) pid=$p]" && numastat -p "$p" 2>/dev/null | tail -4; \
  done ) &
SAMPLER=$!
numactl --interleave=all python -m sglang.bench_one_batch \
  --model-path "$CKPT" \
  --device cpu --tp 1 --batch-size 1 --input-len 1 --output-len 16 \
  --trust-remote-code --disable-cuda-graph \
  --mem-fraction-static 0.05 --max-total-tokens 4096 --swa-full-tokens-ratio 0.5
echo "[exit=$?]"
kill "$SAMPLER" 2>/dev/null
