#!/bin/bash
# Fast CK_DBG probe: per-call C-vs-torch timing for the compressor + sinkhorn regressors, short decode run
# (2 layers, B=32, output-len 8) to localize WHY the C kernels regress in-engine (calls x us/call x C/torch ratio).
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_ckdbg
#SBATCH --output=/scratch/bkaul/dsv4_ckdbg_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_ckdbg_${SLURM_JOB_ID}
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
export INTEL_CPU_DSV4_CK_DBG=1 INTEL_CPU_DSV4_CK_COMPRESSOR=1 INTEL_CPU_DSV4_CK_SINKHORN=1
mkdir -p "$TORCH_EXTENSIONS_DIR"; cd /data/nfs_home/bkaul
INTEL_CPU_DSV4_DECODE_THREADS=32 python $WT/plugin/validate/throughput_bench.py \
  --model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json \
  --tp 1 --batch 32 --input-len 128 --output-len 8 --layers 2 --load-format dummy --mem-fraction 0.85 2>&1 | grep -iE "CK_DBG|decode_tok|prefill_tok"
echo "[done ckdbg]"
