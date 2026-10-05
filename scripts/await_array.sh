#!/bin/bash
# Await a SLURM array job: heartbeat shard states + completed-shard JSON count until the array drains.
# usage: await_array.sh <array_jobid> <shard_json_glob> <expected_shards> [heartbeat_secs]
set +H
AJ="$1"; GLOB="$2"; EXP="${3:-16}"; HB="${4:-180}"
echo "AWAIT_ARRAY jobid=$AJ glob=$GLOB expected=$EXP start=$(date -Is) heartbeat=${HB}s"
elapsed=0
while true; do
  st=$(squeue -j "$AJ" -h -o "%t" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  done_json=$(ls -1 $GLOB 2>/dev/null | wc -l | tr -d ' ')
  if [[ -z "$st" ]]; then
    # array gone from squeue -> confirm once more
    sleep 10
    st2=$(squeue -j "$AJ" -h -o "%t" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
    if [[ -z "$st2" ]]; then
      echo "DONE_ARRAY jobid=$AJ done_json=$(ls -1 $GLOB 2>/dev/null | wc -l | tr -d ' ')/$EXP at=$(date -Is)"
      break
    fi
  fi
  echo "HEARTBEAT jobid=$AJ elapsed=${elapsed}s states=[${st}] done_json=${done_json}/${EXP}"
  sleep "$HB"
  elapsed=$((elapsed+HB))
done
echo "---- shard jsons ----"
ls -1 $GLOB 2>/dev/null
