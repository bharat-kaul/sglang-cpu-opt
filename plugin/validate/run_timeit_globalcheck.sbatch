#!/bin/bash
# Global-vs-local check: does enabling the C compressor/sinkhorn slow NON-DSA ops (MoE/attn) too (=> global
# thread-state perturbation) or only the DSA op (=> local)? Run the plugin's own TIMEIT, CK-off then CK-on,
# fast proxy (2L, mem-frac 0.05, stable extdir). Compare the per-op [DSV4 TIMEIT] breakdown between the two.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --job-name=dsv4_globalchk
#SBATCH --output=/scratch/bkaul/dsv4_globalchk_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_TIMEIT=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=32
cd /data/nfs_home/bkaul
for CK in off on; do
  echo "############ CK=$CK ############"
  ENVX=""; [ "$CK" = on ] && ENVX="INTEL_CPU_DSV4_CK_COMPRESSOR=1 INTEL_CPU_DSV4_CK_SINKHORN=1"
  env $ENVX python $WT/plugin/validate/throughput_bench.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json \
    --tp 1 --batch 32 --input-len 128 --output-len 8 --layers 2 --load-format dummy --mem-fraction 0.05 2>&1 \
    | grep -iE "DSV4 TIMEIT|decode_tok|moe|attn|dsa\.|mhc\.|dense" | tail -40
done
echo "[done globalchk]"
