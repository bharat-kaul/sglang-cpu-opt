#!/bin/bash
# Validate MoE-internal region collapse: MOE_COOP via env (0/1). Correctness (output text identity) + decode timing.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_moecoop
#SBATCH --output=/scratch/bkaul/dsv4_moecoop_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot; CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48" INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000 INTEL_CPU_DSV4_COOP_MHC=1
export INTEL_CPU_DSV4_MOE_COOP=${MOE_COOP:-0}
set +x
RES=$WT/plugin/validate/results/dsv4_moecoop_c${MOE_COOP:-0}_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 python -u $WT/plugin/validate/validate_moe_coop.py \
  --model "$CKPT" --config "$CKPT/config.json" --batch 32 --input-len 128 --output-len 10 --reps 3 \
  --mem-fraction 0.80 --out "$RES" 2>&1 | grep -iE "decode_step|moe_coop|out_texts|Error|Traceback" | tail -12
echo "[done] moecoop=${MOE_COOP} job=$SLURM_JOB_ID"
