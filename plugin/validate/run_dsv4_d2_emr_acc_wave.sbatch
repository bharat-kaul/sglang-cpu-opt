#!/bin/bash
# D2 EMR PARALLEL ACCURACY WAVE with the ALL-C/C++ chain: gsm8k sharded across EMR nodes (correctness is
# ISA-portable + embarrassingly parallel). DSA_CKERNEL=1 routes the DSA chain through the fused C/C++
# kernels; shard 0 additionally runs CHECK=1 (per-call C-vs-eager parity on REAL weights — the correctness
# gate dummy weights can't provide). Confirms the wiring holds accuracy vs the 0.797 baseline.
# Submit: sbatch --array=0-7 run_dsv4_d2_emr_acc_wave.sbatch
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=dsv4_d2acc
#SBATCH --output=/scratch/bkaul/dsv4_d2acc_%A_%a.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
unset INTEL_CPU_DSV4_FP4_MOE_BF16
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=64
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 MAX_RUNNING=64
# ALL-C/C++ chain ON. Shard 0 also asserts per-call real-weight parity (C == eager oracle).
export INTEL_CPU_DSV4_DSA_CKERNEL=1
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then export INTEL_CPU_DSV4_DSA_CKERNEL_CHECK=1; fi
export INTEL_CPU_DSV4_PROV_OUT=/scratch/$USER/dsv4_d2acc_prov_${SLURM_ARRAY_JOB_ID}_sh${SLURM_ARRAY_TASK_ID}.json
CKPT=/scratch/bkaul/models/DeepSeek-V4-Flash
cd /data/nfs_home/bkaul
set +x
set -o pipefail
python -u $WT/plugin/validate/task_gsm8k_chunked.py \
  --model "$CKPT" --data /scratch/$USER/gsm8k_test.jsonl --device cpu --tp 1 \
  --num-questions 320 --num-shots 8 --chunk-size 16 --max-new-tokens 256 \
  --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID --mem-fraction 0.70 --acc-min 0.30 \
  --out $WT/plugin/validate/results/dsv4_d2acc_${SLURM_ARRAY_JOB_ID}_sh${SLURM_ARRAY_TASK_ID}.json \
  --gens-out /scratch/$USER/dsv4_d2acc_${SLURM_ARRAY_JOB_ID}_gens${SLURM_ARRAY_TASK_ID}.json 2>&1 \
  | grep -iE "chunk |acc=|accuracy|verdict|died|OOM|Error|resume|DSA C-kernel|parity"
RC=${PIPESTATUS[0]}   # the PRODUCER's exit (pipefail also makes the pipe fail), not grep's
echo "[exit=${RC}] d2acc shard $SLURM_ARRAY_TASK_ID"
exit $RC              # propagate nonzero child exit to the scheduler (review P0: wrapper failure propagation)
