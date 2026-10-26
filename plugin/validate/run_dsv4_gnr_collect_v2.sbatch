#!/bin/bash
# GNR data collection v2 — FIXES the NUMA-binding pathology that made v1 (job 392789) physically
# invalid (e2e GNR ~6x SLOWER than EMR on compute-bound prefill, impossible for 30.9 TF vs 12.4 TF silicon).
#
# Root cause in v1:
#   - PART 2 (e2e) ran UNBOUND on GNR's 6 SNC NUMA domains -> Linux first-touch piled the entire ~200 GB
#     MXFP4 model onto NUMA node 0, so all domains' cores hammered node-0's memory controller -> BW collapse.
#   - PART 1 (per-kernel) pinned to --cpunodebind=0 but hardcoded OMP_NUM_THREADS=64, oversubscribing a
#     single SNC domain that has far fewer than 64 physical cores.
#
# v2 fixes (topology auto-detected at runtime, no hardcoded counts):
#   - PART 1: OMP_NUM_THREADS = physical cores in NUMA node 0 (no oversubscription), still membind=0.
#   - PART 2: numactl --interleave=all (spread the 200 GB evenly across all domains) + OMP over ALL physical
#     cores with OMP_PLACES=cores. This is the standard "use the whole node sanely" config and matches the
#     EMR node-level comparison (full node, all cores).
#   - Prints an EMR physics reference so the log flags any still-impossible result (GNR must beat EMR prefill).
# Submit: sbatch -p gnrap --job-name dsv4_gnr_collect2 plugin/validate/run_dsv4_gnr_collect_v2.sbatch
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_gnr_collect2
#SBATCH --output=/scratch/bkaul/dsv4_gnr_collect_v2_%j.log
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

# ---- topology auto-detect (physical cores; SMT siblings share the AMX unit, so use physical) ----
TPC=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2);print $2}')
CPS=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')
SOCK=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2);print $2}')
NNODES=$(numactl -H | awk '/^available:/{print $2}')
N0_LOGICAL=$(numactl -H | awk '/^node 0 cpus:/{print NF-3}')
N0_PHYS=$(( N0_LOGICAL / TPC ))
TOT_PHYS=$(( CPS * SOCK ))
echo "[topo] node=$NODE sockets=$SOCK cores/sock=$CPS threads/core=$TPC numa_nodes=$NNODES node0_phys_cores=$N0_PHYS total_phys_cores=$TOT_PHYS"
echo "[ref]  EMR optimized 43L (job 392778, pcl-spr08): prefill 174.5 tok/s, decode 57.5 tok/s."
echo "[ref]  GNR silicon (uarch probe): 30.9 TF AMX bf16 / 2152 GB/s -> GNR MUST beat EMR on compute-bound prefill."

echo "############ PART 1: per-kernel operating-point bench (GNR, 1 SNC domain, NO oversubscription) ############"
( export OMP_NUM_THREADS=$N0_PHYS OMP_PROC_BIND=close OMP_PLACES=cores
  export OP_OUT=$RDIR/operating_point_gnr_v2.json
  echo "[part1] OMP_NUM_THREADS=$OMP_NUM_THREADS (node0 physical cores), membind=0"
  numactl --cpunodebind=0 --membind=0 python $WT/plugin/validate/bench_operating_point.py 2>&1 | tail -30 )

echo "############ PART 2: e2e wall-time matrix (GNR), baseline vs optimized x {4L,8L} — INTERLEAVED memory ############"
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export INTEL_CPU_DSV4_REAL_ROUTER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5

run_leg () {  # $1=mode $2=layers
  ( local mode=$1 NL=$2
    if [ "$mode" = "optimized" ]; then
      export INTEL_CPU_DSV4_DSA_CKERNEL=1
    else
      export INTEL_CPU_DSV4_DSA_CKERNEL=0
    fi
    unset OMP_WAIT_POLICY KMP_BLOCKTIME INTEL_CPU_DSV4_DECODE_THREADS
    # throughput-optimal: spin OMP over ALL physical cores; weights interleaved across all NUMA domains
    export OMP_NUM_THREADS=$TOT_PHYS OMP_PROC_BIND=close OMP_PLACES=cores
    export INTEL_CPU_DSV4_PROV_OUT=$RDIR/perfmatrix_v2_${J}_${NODE}_${mode}_${NL}L_prov.json
    echo "===== LEG mode=$mode layers=$NL node=$NODE OMP=$TOT_PHYS interleave=all ====="
    numactl --interleave=all python -u $WT/plugin/validate/throughput_bench.py \
      --model "$CKPT" --config "$CKPT/config.json" --tp 1 --batch 32 --input-len 128 --output-len 8 \
      --layers $NL --load-format dummy --mem-fraction 0.70 \
      --out $RDIR/perfmatrix_v2_${J}_${NODE}_${mode}_${NL}L.json 2>&1 \
      | grep -iE "prefill_tok|decode_tok|injected|fell back|Error|Traceback|OOM" | tail -12
  )
}
for NL in 4 8; do run_leg baseline $NL; run_leg optimized $NL; done
echo "[done] GNR collect v2 node=$NODE job=$J"
