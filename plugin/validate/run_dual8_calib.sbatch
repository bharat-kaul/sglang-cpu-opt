#!/bin/bash
# Dual-NUMA (full-node, both sockets, no bind) 8L B=32 decode -> completes the DUAL-NUMA 4+8 calibration
# (4L dual = 90.8 tok/s from the numa tiebreaker) so the 43L B=32 full-node decode headline extrapolates.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=dsv4_dual8
#SBATCH --output=/scratch/bkaul/dsv4_dual8_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=32 OMP_NUM_THREADS=128
cd /data/nfs_home/bkaul
for NL in 4 8; do
  echo "############ dual-NUMA ${NL}L ############"
  python $WT/plugin/validate/throughput_bench.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json \
    --tp 1 --batch 32 --input-len 128 --output-len 8 --layers $NL --load-format dummy --mem-fraction 0.85 2>&1 \
    | grep -iE "prefill_tok|decode_tok"
done
echo "[done dual8]"
