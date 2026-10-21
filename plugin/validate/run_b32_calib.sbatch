#!/bin/bash
# B=32 two-depth calibration (proxy4 + proxy8) at the OPERATING POINT (S=1024), DSA-CPU route enabled so
# B>=8 does NOT OOM on the materialized-attention path. Runs BOTH depths in ONE job on the SAME node+config
# so t_layer=(t8-t4)/4 and t_fixed=t4-4*t_layer are valid for the B=32 regime -> correct 43L decode-headline
# extrapolation (the existing calibration only has B=1). B=1 included as a cross-check/floor.
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=dsv4_b32calib
#SBATCH --output=/scratch/bkaul/dsv4_b32calib_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
WT=/data/nfs_home/bkaul/dsv4-tree-pilot
export SGLANG_USE_CPU_ENGINE=1 INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1
export DETERMINISTIC_DUMMY=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH=$WT/plugin
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_dsv4_stable
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 INTEL_CPU_DSV4_DECODE_THREADS=32
cd "$WT"
INLEN=${INLEN:-128}; OUTLEN=${OUTLEN:-8}; BATCHES=${BATCHES:-1,32}
echo "node=$(hostname) git=$(git rev-parse --short HEAD)"
for TAG in proxy4 proxy8; do
  MODEL=/scratch/$USER/models/flash-v4.1-$TAG
  NL=$(python -c "import json;print(json.load(open('$MODEL/config.json'))['num_hidden_layers'])")
  echo ">>> ===== $TAG ($NL layers) B=$BATCHES S=$INLEN ====="
  OMP_NUM_THREADS=64 numactl --cpunodebind=0 --membind=0 \
    python $WT/plugin/validate/bench_wall_time.py \
      --model "$MODEL" --batch-list "$BATCHES" --input-len "$INLEN" --output-len "$OUTLEN" \
      --reps 3 --tp 1 --dtype bfloat16 --load-format dummy --mem-fraction 0.85 \
      --out /scratch/$USER/dsv4_b32calib_${TAG}_${SLURM_JOB_ID}.json
  echo "[exit=${PIPESTATUS[0]}] $TAG"
done
echo "[done b32calib]"
