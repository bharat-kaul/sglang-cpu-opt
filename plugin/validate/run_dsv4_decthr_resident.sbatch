#!/bin/bash
# LEVER 1 (resident) — decode thread sweep on the REAL model, ONE load, SINGLE SOCKET, physical cores only.
# Supersedes job 393786 (11 reloads + OMP_PROC_BIND/PLACES thread-COLLAPSE -> flat 4 tok/s = invalid).
# Pin to 58 socket-0 PHYSICAL cores (0-57; leaves 2 for main/OS, the bound-2 headroom; SMT 120-179 and
# socket 1 excluded), membind=0, passive wait set FIRST, and DO NOT set OMP_PROC_BIND/OMP_PLACES (that
# collapses SGLang to 1 thread). Sweep decode threads via the control file (plugin logs the HELD count).
# Submit: sbatch -p emr --job-name dsv4_decthr plugin/validate/run_dsv4_decthr_resident.sbatch
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dsv4_decthr
#SBATCH --output=/scratch/bkaul/dsv4_decthr_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
CTL=/scratch/bkaul/dsv4_decthr_ctl_${SLURM_JOB_ID}.txt
echo 58 > "$CTL"
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS_FILE="$CTL"      # resident sweep: plugin re-reads per decode forward
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0        # park idle threads (set FIRST). NO proc_bind/places (collapse).
set +x
RES=$WT/plugin/validate/results/dsv4_decthr_${SLURM_JOB_ID}.json
# single socket, physical cores only: 0-57 (58 of 60; SMT 120-179 + socket1 60-119/180-239 excluded)
numactl --physcpubind=0-57 --membind=0 \
  python -u $WT/plugin/validate/decode_thread_sweep_resident.py \
    --model "$CKPT" --config "$CKPT/config.json" --ctl "$CTL" \
    --threads 8,16,30,45,58 --batches 1,32 --input-len 8 --output-len 4 --reps 2 \
    --mem-fraction 0.80 --out "$RES" 2>&1 \
  | grep -iE "\[sweep\]|\[OPT\]|DECODE THREADCAP|Error|Traceback|OOM" | tail -60
echo "############ RESULT ############"; cat "$RES" 2>/dev/null
echo "[done] resident decode thread sweep job=$SLURM_JOB_ID"
