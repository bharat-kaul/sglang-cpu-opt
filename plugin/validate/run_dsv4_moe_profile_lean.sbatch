#!/bin/bash
# PROFILING PASS: where does the B=32 MoE-apply 100->170 GB/s gap go? Run a real B=32 decode under perf record
# with the SYMBOL-BEARING local build (shadow package, no venv clobber), then break down self-time inside the
# kernel (tinygemm_kernel_nn / unpack / copy_stub / silu / scale). Single socket, warm, 32 threads.
#   sbatch -p emr --job-name moe_prof plugin/validate/run_dsv4_moe_profile.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/bkaul/dsv4_moe_prof_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
# shadow sgl_kernel (my symbol-bearing build) FIRST, then plugin
export PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=32 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
set +x
# confirm the shadow build is the one loaded
python3 -c "import sgl_kernel; print('[profile] common_ops:', sgl_kernel.common_ops.__file__)"
DATA=/scratch/bkaul/dsv4_moe_prof_${SLURM_JOB_ID}.perf
# perf record the whole decode-heavy run (freq 999, call graph dwarf for inlined kernel frames)
numactl --physcpubind=0-57 --membind=0 \
  perf record -F 499 -o $DATA -- \
  python -u $WT/plugin/validate/validate_decode_threads_clean.py \
    --model "$CKPT" --config "$CKPT/config.json" \
    --batches 32 --input-len 8 --output-len 20 --reps 3 --mem-fraction 0.80 2>&1 \
  | grep -iE "\[CLEAN\]|profile|Error|Traceback|OOM" | tail -20
echo "########## PERF REPORT: kernel self-time breakdown ##########"
perf report -i $DATA --stdio --no-children --percent-limit 0.5 2>/dev/null \
  | grep -iE "tinygemm|unpack|copy_stub|silu|cvt_mxfp4|fused_experts|convert_weight|add_bias|copy_mul|brgemm|dpbf16|moe" \
  | head -40
echo "########## TOP 30 SYMBOLS OVERALL ##########"
perf report -i $DATA --stdio --no-children 2>/dev/null | grep -vE "^#" | awk 'NF>=5' | head -30
echo "[done] moe profile job=$SLURM_JOB_ID"
