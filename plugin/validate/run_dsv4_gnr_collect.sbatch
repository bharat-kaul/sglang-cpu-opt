#!/bin/bash
# GNR data collection in ONE node grab: (1) e2e wall-time matrix (baseline vs optimized x 4L/8L) for
# the node-walltime chart, AND (2) per-kernel operating-point bench for the kernel-pilot GNR-vs-EMR table.
# Submit: sbatch -p gnrap plugin/validate/run_dsv4_gnr_collect.sbatch
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_gnr_collect
#SBATCH --output=/scratch/bkaul/dsv4_gnr_collect_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
RDIR=$WT/plugin/validate/results
J=${SLURM_JOB_ID}
NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
export PYTHONPATH=$WT/plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
cd "$WT"
set +x

echo "############ PART 1: per-kernel operating-point bench (GNR) ############"
( export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
  export OP_OUT=$RDIR/operating_point_gnr.json
  numactl --cpunodebind=0 --membind=0 python $WT/plugin/validate/bench_operating_point.py 2>&1 | tail -25 )

echo "############ PART 2: e2e wall-time matrix (GNR), baseline vs optimized x {4L,8L} ############"
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_REAL_ROUTER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5

run_leg () {  # $1=mode $2=layers
  ( local mode=$1 NL=$2
    if [ "$mode" = "optimized" ]; then
      export INTEL_CPU_DSV4_DSA_CKERNEL=1; unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
    else
      export INTEL_CPU_DSV4_DSA_CKERNEL=0; unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
    fi
    export INTEL_CPU_DSV4_PROV_OUT=$RDIR/perfmatrix_${J}_${NODE}_${mode}_${NL}L_prov.json
    echo "===== LEG mode=$mode layers=$NL node=$NODE ====="
    python -u $WT/plugin/validate/throughput_bench.py \
      --model "$CKPT" --config "$CKPT/config.json" --tp 1 --batch 32 --input-len 128 --output-len 8 \
      --layers $NL --load-format dummy --mem-fraction 0.70 \
      --out $RDIR/perfmatrix_${J}_${NODE}_${mode}_${NL}L.json 2>&1 \
      | grep -iE "prefill_tok|decode_tok|injected|fell back|Error|Traceback|OOM" | tail -12
  )
}
for NL in 4 8; do run_leg baseline $NL; run_leg optimized $NL; done
echo "[done] GNR collect node=$NODE job=$J"
