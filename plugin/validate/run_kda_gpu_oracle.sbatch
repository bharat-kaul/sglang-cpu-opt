#!/bin/bash
# KDA GPU-oracle parity (1 GPU): CPU reference recurrence vs the real Triton
# fused_recurrent_kda. Independent-spec validation of the sole authoring gap.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=glm_kda_oracle
#SBATCH --output=/scratch/bkaul/glm_kda_oracle_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/test_kda_gpu_oracle.py
echo "[exit=$?]"
