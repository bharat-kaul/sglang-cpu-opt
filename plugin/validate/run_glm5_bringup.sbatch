#!/bin/bash
# GLM-5.3 Flash CPU bring-up ladder: tiny arch-faithful config + DUMMY (bf16) weights.
# Shakes out the KDA/MHC/MLA/DSA/MoE wiring + the mamba dual-cache + MLA paged-pool infra
# WITHOUT the big node or the fp8 bridge. Each break -> fix in the plugin -> re-run.
# Portable dummy-weight correctness run -> submit to WHICHEVER of emr/gnrap frees first
# (Slurm picks the partition that can start it soonest). GNR is fine for a tiny bring-up.
#SBATCH --partition=emr,gnrap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --job-name=glm5_bringup
#SBATCH --output=/scratch/bkaul/glm5_bringup_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/make_glm5_tiny.py
export MODEL_PATH=/scratch/bkaul/models/glm5-tiny
export LOAD_FORMAT=dummy
export MEM_FRAC=0.30
export LOG_LEVEL=info
# CPU-valid mamba radix cache (KDA linear-attn): no_buffer (extra_buffer needs GPU-FLA).
# page_size left to the MLA/DSA resolver (64); the CPU no_buffer assert is relaxed in the plugin.
export MAMBA_RADIX_STRATEGY=no_buffer
# Disable radix prefix-caching -> ChunkCache (no MambaComponent page_size=1 clash with DSA's 64).
export DISABLE_RADIX=1
# PREFILL-ONLY (max_new=1): the correctness gate is the per-layer PREFILL fingerprint; prefill
# works end-to-end. Decode (NoPE-MLA fused-rope CPU path) is a separate deferred task.
export MAX_NEW=1
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py
echo "[exit=$?]"
