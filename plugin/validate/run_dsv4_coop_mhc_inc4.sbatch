#!/bin/bash
# Monolith increment 4 in-model validation: coop hc_combine+norm fusion. (1) CHECK run = in-model parity,
# (2) coop-off vs coop-on decode A/B (fresh processes, env set pre-spawn = valid), shadow symbol build.
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bkaul/dsv4_coop_mhc_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=32 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
set +x
python3 -c "import sgl_kernel,torch; print('[coop] common_ops:', sgl_kernel.common_ops.__file__); print('[coop] op present:', hasattr(torch.ops.sgl_kernel,'coop_hc_combine_rmsnorm_f32_cpu'))"

echo "########## (1) IN-MODEL PARITY: COOP_MHC=1 + CHECK=1 (any parity break raises) ##########"
INTEL_CPU_DSV4_COOP_MHC=1 INTEL_CPU_DSV4_DSA_CKERNEL_CHECK=1 numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" --batches 32 --input-len 8 --output-len 6 --reps 1 \
    --mem-fraction 0.80 2>&1 | grep -iE "\[CLEAN\]|coop|parity|assert|Error|Traceback|fell back" | tail -12

for C in 0 1; do
  echo "########## (2) A/B: COOP_MHC=$C decode ##########"
  INTEL_CPU_DSV4_COOP_MHC=$C numactl --physcpubind=0-57 --membind=0 \
    python -u $WT/plugin/validate/validate_decode_threads_clean.py \
      --model "$CKPT" --config "$CKPT/config.json" --batches 1,32 --input-len 8 --output-len 8 --reps 4 \
      --mem-fraction 0.80 --out $WT/plugin/validate/results/dsv4_coop_mhc_c${C}_${SLURM_JOB_ID}.json 2>&1 \
    | grep -iE "\[CLEAN\]|fell back|Error|Traceback" | tail -6
done
echo "[done] coop mhc inc4 job=$SLURM_JOB_ID"
