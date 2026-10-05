#!/bin/bash
# Auto sneak-preview FAIL-FAST watcher for a sharded, chunk-checkpointed accuracy run.
# Arms the codified sneak-preview as an AUTOMATIC, at-launch check (not a reactive, run-it-when-asked
# step): periodically COMBINEs the partial shard jsons -> running accuracy, and ALERTS (or optionally
# scancels) when the running accuracy falls clearly below an expected floor after a minimum sample, so
# a regressed harness/model is caught within ~1 chunk instead of burning the whole wall.
#
# Launch it ASYNC right after submitting the array (the sneak-preview analogue of await_job.sh):
#   bash scripts/sneak_preview.sh <array_jobid> <shard_json_glob> [floor] [min_q] [interval_s] [--abort]
# Default is ALERT-ONLY (safe); pass --abort to auto-`scancel` the array on a clear regression.
set +H
AJ="$1"; GLOB="$2"; FLOOR="${3:-0.80}"; MINQ="${4:-160}"; IV="${5:-300}"; ABORT="${6:-}"
COMBINE="python3 $(cd "$(dirname "$0")/.." && pwd)/plugin/validate/combine_gsm8k_shards.py"
echo "SNEAK_PREVIEW jobid=$AJ glob=$GLOB floor=$FLOOR min_q=$MINQ interval=${IV}s abort=${ABORT:-off} start=$(date -Is)"
alerted=0
while true; do
  states=$(squeue -j "$AJ" -h -o "%t" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  read q c inv acc <<<"$($COMBINE --glob "$GLOB" 2>/dev/null | python3 -c "import json,sys
try:
  d=json.load(sys.stdin); print(d['questions_scored'], d['correct'], d['invalid'], d['accuracy'])
except Exception: print('0 0 0 0')")"
  echo "SNEAK jobid=$AJ t=$(date +%H:%M:%S) scored=${q} correct=${c} invalid=${inv} acc=${acc} states=[${states}]"
  if [ "${q:-0}" -ge "$MINQ" ] && [ "$alerted" -eq 0 ]; then
    below=$(python3 -c "print(1 if float('${acc:-0}') < float('$FLOOR') else 0)")
    invhi=$(python3 -c "q=${q:-0}; print(1 if q>0 and ${inv:-0}/q > 0.05 else 0)")
    if [ "$below" = "1" ] || [ "$invhi" = "1" ]; then
      echo "ALERT jobid=$AJ FAIL-FAST tripped: acc=${acc} floor=${FLOOR} invalid=${inv}/${q} — a regressed harness/model should be FIXED + relaunched, not left to burn the wall."
      alerted=1
      if [ "$ABORT" = "--abort" ]; then echo "ABORT: scancel $AJ"; scancel "$AJ"; break; fi
    else
      echo "OK jobid=$AJ sneak-preview PASS: acc=${acc} >= floor=${FLOOR}, invalid rate low (scored=${q}) — let it ride."
    fi
  fi
  if [ -z "$states" ]; then echo "SNEAK_DONE jobid=$AJ final: scored=${q} correct=${c} invalid=${inv} acc=${acc} at=$(date -Is)"; break; fi
  sleep "$IV"
done
