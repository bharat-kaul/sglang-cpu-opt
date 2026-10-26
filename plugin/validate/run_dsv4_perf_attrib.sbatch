#!/bin/bash
# ONE-LEVER-AT-A-TIME attribution for the decode confound: toggle DSA kernels and OMP policy
# independently so the baseline-vs-optimized decode delta is attributable, not guessed.
#   A naive      : eager DSA + OMP spin   (fully naive baseline)
#   B kernels    : C DSA    + OMP spin    (isolate the C-kernel contribution, OMP held fixed)
#   C kern+passiv: C DSA    + OMP passive (add the OMP-passive policy on top)
# All at B=32 throughput, all cores, real-router ON. 4L+8L for the two-depth model.
# Submit: sbatch -p emr --job-name dsv4_attr plugin/validate/run_dsv4_perf_attrib.sbatch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/bkaul/dsv4_perfattrib_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_REAL_ROUTER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
cd /data/nfs_home/bkaul
set +x

leg () {  # $1=tag $2=ck $3=omp(spin|passive) $4=layers
  ( local tag=$1 ck=$2 omp=$3 NL=$4
    export INTEL_CPU_DSV4_DSA_CKERNEL=$ck
    if [ "$omp" = "passive" ]; then export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
    else unset OMP_WAIT_POLICY KMP_BLOCKTIME; fi
    unset INTEL_CPU_DSV4_DECODE_THREADS
    echo "===== $tag (ck=$ck omp=$omp) ${NL}L ====="
    stdbuf -oL python -u $WT/plugin/validate/throughput_bench.py \
      --model "$CKPT" --config "$CKPT/config.json" --tp 1 \
      --batch 32 --input-len 128 --output-len 8 --layers $NL \
      --load-format dummy --mem-fraction 0.70 \
      --out $RDIR/perfattrib_${J}_${NODE}_${tag}_${NL}L.json 2>&1 \
      | grep -iE "prefill_tok|decode_tok|fell back|Error|Traceback|OOM" | tail -8
  )
}

for NL in 4 8; do
  leg A_naive        0 spin    $NL
  leg B_kernels      1 spin    $NL
  leg C_kern_passive 1 passive $NL
done
echo "[done] attrib node=$NODE job=$J"
