#!/bin/bash
# DECODE overhead-attribution at the EMR operating point (held-fixed A/B, 4L dummy proxy).
# Runs the plugin TIMEIT boundary timers WITH phase separation (prefill vs decode) and dumps the
# machine-readable per-op framework/kernel/dispatch split to JSON, CK-off then CK-on, EVERYTHING ELSE
# HELD FIXED (same job, 4L, B=32, mem-frac 0.85 -- the job-386039 config). Gives the authoritative
# decode framework/dispatch share (replaces the unsourced "~39%") so the overhead pass targets the
# right bucket. Depth-invariant share -> extrapolates to 43L.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_ohattr
#SBATCH --output=/scratch/bkaul/dsv4_ohattr_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_TIMEIT=1 INTEL_CPU_DSV4_TIMEIT_PHASE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=32
cd /data/nfs_home/bkaul
for CK in off on; do
  echo "############ CK=$CK ############"
  ENVX=""; [ "$CK" = on ] && ENVX="INTEL_CPU_DSV4_DSA_CKERNEL=1"
  env $ENVX INTEL_CPU_DSV4_TIMEIT_JSON=/scratch/$USER/dsv4_ohattr_${SLURM_JOB_ID}_${CK}.json \
    python $WT/plugin/validate/throughput_bench.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --config /scratch/bkaul/models/DeepSeek-V4-Flash/config.json \
    --tp 1 --batch 32 --input-len 128 --output-len 8 --layers 4 --load-format dummy --mem-fraction 0.85 2>&1 \
    | grep -iE "DSV4 TIMEIT|overhead split|prefill_tok|decode_tok| moe| attn|dsa\.|mhc\.|dense|%  n=|%  [a-z]"
done
echo "[done ohattr]"
