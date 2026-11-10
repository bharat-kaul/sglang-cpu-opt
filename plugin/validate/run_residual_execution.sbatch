#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=dsv4_residual
#SBATCH --output=/scratch/bkaul/dsv4_residual_%j.log
set -euo pipefail

# Positional inputs: frozen source checkout, evidence root, arm name, prefill size, shared clamp.
WT=$(realpath "${1:?source checkout required}")
ROOT=$(realpath "${2:?evidence root required}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ROOT=$(realpath "$ROOT/shard_${SLURM_ARRAY_TASK_ID}")
fi
ARM=${3:?arm required}
PREFILL=${4:?prefill size required}
SHARED=${5:?shared clamp required}
case "$ARM:$PREFILL:$SHARED" in
    baseline:512:0|shared:512:1|whole:-1:0|whole_shared:-1:1|quant:512:1) ;;
    *) echo "unsupported experiment arm: $ARM:$PREFILL:$SHARED" >&2; exit 2 ;;
esac

source /scratch/bkaul/venvs/sglang-cpu/bin/activate
export PYTHONUNBUFFERED=1
export SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models PYTHONPATH="$WT/plugin"
export INTEL_CPU_DSV4_DSA_CPU=1 INTEL_CPU_DSV4_MXFP4_MOE=1 INTEL_CPU_DSV4_DSA_CKERNEL=1
export INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1
export INTEL_CPU_DSV4_SHARED_CLAMP="$SHARED"
export INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=0
if [[ "$ARM" == "quant" ]]; then
    export INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC=1
fi
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5
export INTEL_CPU_DSV4_DECODE_THREADS=8 INTEL_CPU_DSV4_DECODE_THREADS_STICKY=1
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 INTEL_CPU_DSV4_COOP_MHC=0
OUT="$ROOT/${ARM}_${SLURM_JOB_ID}"
mkdir "$OUT"
export TORCH_EXTENSIONS_DIR="$OUT/extensions" MAX_JOBS=2
export INTEL_CPU_DSV4_EXECUTION_TRACE="$OUT/forward"
if [[ -f "$ROOT/capture_plan.json" ]]; then
    export DSV4_SHARED_REGRESSION_PLAN="$ROOT/capture_plan.json"
    export DSV4_SHARED_REGRESSION_OUT="$OUT/shared_capture"
fi
{
    echo "job=$SLURM_JOB_ID node=$(hostname) arm=$ARM prefill=$PREFILL shared=$SHARED"
    echo "source=$WT commit=$(git -C "$WT" rev-parse HEAD)"
    echo "protocol=production-text shots=8 maxnew=256 chunk_size=32 dtype=bfloat16 mem_fraction=0.5"
    env | LC_ALL=C sort | grep -E '^(INTEL_CPU_DSV4_|CONTEXT_LEN=|MAX_TOTAL_TOKENS=|SWA_RATIO=|OMP_|KMP_|GOMP_|SGLANG_)'
    sha256sum "$ROOT/question_ids.json" "$WT/plugin/validate/task_gsm8k_chunked.py" \
        "$WT/plugin/intel_cpu_models/_dsv4_cpu_infra.py" \
        "$WT/plugin/dsv4_layer_capture/execution_trace.py" \
        "$WT/plugin/intel_cpu_models/_dsv4_moe_clamp.py" \
        "$WT/plugin/intel_cpu_models/_dsv4_shared_clamp.py" \
        "$WT/plugin/kernels/dsv4_shared_quant_diagnostic.h"
} > "$OUT/stamp.txt"
cat "$OUT/stamp.txt"
cd "$WT/plugin/validate"
python task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --num-questions 1319 --question-ids-file "$ROOT/question_ids.json" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --chunk-size 32 --num-shots 8 --max-new-tokens 256 --dtype bfloat16 --mem-fraction 0.5 \
    --watchdog-timeout 7000 --chunked-prefill-size "$PREFILL" --no-resume \
    --out "$OUT/score.json" --gens-out "$OUT/gens.json" --gens-schema-v2 --gens-token-ids \
    --runtime-out "$OUT/runtime.json"
if [[ -f "$ROOT/capture_plan.json" ]]; then
    python replay_shared_regression.py "$OUT/shared_capture" "$ROOT/capture_plan.json"
fi
echo "RESIDUAL_EXECUTION_COMPLETE arm=$ARM job=$SLURM_JOB_ID out=$OUT"
