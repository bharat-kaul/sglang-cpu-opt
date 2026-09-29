#!/bin/bash
# GPU REFERENCE: native sglang CUDA path (NO plugin, NO CPU env) as the ground-truth oracle.
# Coherent output => the CPU bug is integration-level (checkpoint is fine); still garbage =>
# the checkpoint/model itself is suspect. Partial GPUs (tp=4) to avoid stalling on the whole node.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_gpu_ref
#SBATCH --output=/scratch/bkaul/dsv4_gpu_ref_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
# NOTE: no SGLANG_USE_CPU_ENGINE, no SGLANG_EXTERNAL_MODEL_PACKAGE, no INTEL_CPU_* -> pure native GPU.
export ACC_DEVICE=cuda
export ACC_TP=4
export ACC_QUICK=0
export ACC_MAX_NEW=8
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/accuracy_generate.py
echo "[exit=$?]"
