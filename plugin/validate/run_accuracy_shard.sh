#!/bin/bash
set -euo pipefail
ROOT=$(realpath "${1:?evidence root}")
DEVICE=${2:?cpu or cuda}
SHARD=${3:?shard index}
TP=${4:?tensor parallel size}
case "$DEVICE" in cpu) ARM=cpu; FRACTION=0.5 ;; cuda) ARM=gpu; FRACTION=0.85 ;; *) exit 2 ;; esac
OUT="$ROOT/$ARM/shard_$SHARD"
mkdir -p "$OUT"
export CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=131072 SWA_RATIO=0.5 DISABLE_RADIX=1
{
    echo "job=$SLURM_JOB_ID node=$(hostname) device=$DEVICE tp=$TP shard=$SHARD"
    echo "protocol=8shot greedy maxnew256 stops_Question_Assistant batch32 prefill512 shards16 radix_off"
    sha256sum "$ROOT/source_hashes.json" "$ROOT/plan.json" /scratch/bkaul/gsm8k_test.jsonl
    env | LC_ALL=C sort | grep -E '^(INTEL_CPU_DSV4_|SGLANG_|OMP_|KMP_|GOMP_|CONTEXT_LEN=|MAX_TOTAL_TOKENS=|SWA_RATIO=|DISABLE_RADIX=)'
} > "$OUT/stamp.txt"
python -u "$ROOT/source/plugin/validate/task_gsm8k_chunked.py" \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash --data /scratch/bkaul/gsm8k_test.jsonl \
    --device "$DEVICE" --tp "$TP" --dtype bfloat16 --mem-fraction "$FRACTION" \
    --num-questions 1319 --num-shots 8 --num-shards 16 --shard-index "$SHARD" \
    --chunk-size 32 --max-new-tokens 256 --chunked-prefill-size 512 \
    --watchdog-timeout 7000 --no-resume \
    --out "$OUT/score.json" --gens-out "$OUT/gens.json" --gens-schema-v2 --gens-token-ids \
    --runtime-out "$OUT/runtime.json"
touch "$OUT/complete"
echo "ACCURACY_SHARD_COMPLETE device=$DEVICE shard=$SHARD job=$SLURM_JOB_ID"
