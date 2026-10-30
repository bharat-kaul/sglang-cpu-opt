#!/bin/bash
# END-TASK ACCURACY at the SHIPPED optimized config: warm pool + M-adaptive threads + coop-MHC together.
# Certifies accuracy is maintained for the performance numbers (rule P: perf run uses CHECK=0, so prove
# accuracy with a separate matched-config end-task run).
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_fullopt_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
SHADOW=/scratch/bkaul/sgl_kernel_shadow
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$SHADOW:$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
# SHIPPED optimized config:
export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48" INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
export INTEL_CPU_DSV4_COOP_MHC=1
set +x
python3 -c "import sgl_kernel,torch; print('[acc] common_ops:', sgl_kernel.common_ops.__file__); print('[acc] coop present:', hasattr(torch.ops.sgl_kernel,'coop_hc_post_cpu'))"
RES=$WT/plugin/validate/results/dsv4_gsm8k_fullopt_${SLURM_JOB_ID}.json
numactl --physcpubind=0-57 --membind=0 \
  python $WT/plugin/validate/task_gsm8k.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions ${NQ:-200} --num-shots 8 --max-new-tokens 256 \
    --watchdog-timeout 8400 --chunked-prefill-size 512 --out "$RES" 2>&1 \
  | grep -iE "accuracy|correct|questions|coop|fell back|Error|Traceback" | tail -12
echo "####### RESULT #######"; cat "$RES" 2>/dev/null
echo "[done] gsm8k fullopt job=$SLURM_JOB_ID"
