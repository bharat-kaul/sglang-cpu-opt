#!/bin/bash
# DSV4 torch-bucket sizing: dispatch-count + thread-sweep + ctx-sweep on the real DSA ops (+ sinkhorn),
# config-accurate batch-1 decode shapes. Separates dispatch-overhead vs fp32-compute vs thread-oversubscription.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --job-name=dsv4_sizing
#SBATCH --output=/scratch/bkaul/dsv4_sizing_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
cd /data/nfs_home/bkaul/sglang-cpu-opt
OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 numactl --cpunodebind=0 --membind=0 \
  env ITERS=50 S_LIST=512,2048,8192 THREADS=1,8,16,32,64,128 \
      python plugin/validate/dsv4_torch_sizing.py
echo "[exit=$?]"
