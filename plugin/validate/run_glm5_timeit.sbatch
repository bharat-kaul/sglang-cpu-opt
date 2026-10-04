#!/bin/bash
# GLM-5.3 Flash PERF PROXY profile: depth-reduced (NUM_LAYERS, keep FULL width) + dummy weights +
# per-op TIMEIT. Prefill(256 tok) + decode(MAX_NEW steps), phase-split. EMR (AMX, parallel finding;
# perf NOT ISA-portable -> final numbers re-run on gnrap). Emits the pivot JSON for
# time_attribution_pivot.py. NOT a correctness run (dummy weights; perf is valid on real shapes/dtypes).
# N=8 covers every op family: KDA(linear) L0-2/4-6 + MLA(sparse) L3/L7 + dense MLP L0-2 + MoE L3-7 + MHC.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=glm5_timeit
#SBATCH --output=/scratch/bkaul/glm5_timeit_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
cd /data/nfs_home/bkaul
export MODEL_PATH=/scratch/bkaul/models/GLM-5.3-Flash
export LOAD_FORMAT=dummy
export NUM_LAYERS=${NUM_LAYERS:-8}
export MEM_FRAC=${MEM_FRAC:-0.8}
export MAX_RUNNING=1
export CONTEXT_LEN=${CONTEXT_LEN:-4096}
export DISABLE_RADIX=1
export MAMBA_RADIX_STRATEGY=no_buffer
export MAX_NEW=${MAX_NEW:-8}
# 256-token synthetic prefill (values irrelevant with dummy weights; shapes/byte-streaming are real).
export TOKEN_IDS=$(python3 -c "print(','.join(str(i) for i in range(1,257)))")
# Per-op TIMEIT (reuses the shared DSV4 timer; GLM wraps layer+KDA/MLA/MoE/dense/hc_post).
export INTEL_CPU_DSV4_TIMEIT=1
export INTEL_CPU_DSV4_TIMEIT_PHASE=1
# MoE thread-cliff sweep: cap the threads around the native fused_experts_cpu apply (DSV4 found it
# scales INVERSELY with threads). Empty = no cap (baseline, uses all NUMA-domain cores).
export INTEL_CPU_DSV4_MOE_THREADS=${MOE_THREADS:-}
# Explicitly re-export (not just --export passthrough) so the scheduler subprocess reliably sees it.
export INTEL_CPU_GLM_FAST_ROUTER=${INTEL_CPU_GLM_FAST_ROUTER:-}
export INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/bkaul/glm5_timeit_${SLURM_JOB_ID}.json
export TIMEIT_MODEL="GLM-5.3-Flash"
export TIMEIT_NODE="${TIMEIT_NODE:-EMR}"
export TIMEIT_PRECISION="fp8 e4m3 block W8A16, bf16 compute"
export TIMEIT_BATCH=1
export LOG_LEVEL=warning
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py
echo "[timeit json] $INTEL_CPU_DSV4_TIMEIT_JSON"; ls -la "$INTEL_CPU_DSV4_TIMEIT_JSON" 2>/dev/null
echo "[exit=$?]"
