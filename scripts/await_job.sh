#!/bin/bash
# await_job.sh <jobid> [logfile] [heartbeat_secs]
#
# Block on a Slurm job and emit a periodic HEARTBEAT (elapsed + last log line), then a
# final summary (state + exit code + log tail) when it reaches a terminal state.
#
# USAGE PATTERN (the whole point): launch this via the agent's ASYNC terminal so the
# agent is AUTO-NOTIFIED on completion instead of a human having to prompt "is it done?".
# The heartbeat lines make liveness visible; the terminal-state exit triggers the wake.
set -u
JID="${1:?usage: await_job.sh <jobid> [logfile] [hb_secs]}"
LOG="${2:-}"
HB="${3:-60}"
[ -z "$LOG" ] && LOG=$(scontrol show job "$JID" 2>/dev/null | sed -n 's/^.*StdOut=//p' | head -1)
START=$(date +%s)
echo "AWAIT jobid=$JID log=$LOG start=$(date -Is) heartbeat=${HB}s"
while :; do
  ST=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
  if [ -z "$ST" ]; then
    FINAL=$(sacct -j "$JID" --format=State -nX 2>/dev/null | head -1 | tr -d ' ')
    EC=$(sacct -j "$JID" --format=ExitCode -nX 2>/dev/null | head -1 | tr -d ' ')
    echo "DONE jobid=$JID state=${FINAL:-UNKNOWN} exit=${EC:-?} at=$(date -Is)"
    echo "---- log tail ----"; tail -25 "$LOG" 2>/dev/null || echo "(no log at $LOG)"
    break
  fi
  EL=$(( $(date +%s) - START ))
  LAST=$(tail -1 "$LOG" 2>/dev/null | cut -c1-80)
  echo "HEARTBEAT jobid=$JID state=$ST elapsed=${EL}s last='${LAST}'"
  sleep "$HB"
done
