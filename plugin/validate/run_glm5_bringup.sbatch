#!/bin/bash
# GLM-5.3 Flash CPU bring-up ladder: tiny arch-faithful config + DUMMY (bf16) weights.
# Shakes out the KDA/MHC/MLA/DSA/MoE wiring + the mamba dual-cache + MLA paged-pool infra
# WITHOUT the big node or the fp8 bridge. Each break -> fix in the plugin -> re-run.
#SBATCH --partition=emr
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
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py
echo "[exit=$?]"
