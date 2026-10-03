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
EMPTY=0
echo "AWAIT jobid=$JID log=$LOG start=$(date -Is) heartbeat=${HB}s"
while :; do
  ST=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
  if [ -z "$ST" ]; then
    # squeue empty can be a TRANSIENT blip (or a crash) — confirm with sacct's
    # authoritative terminal state before concluding DONE, else keep waiting.
    FINAL=$(sacct -j "$JID" --format=State -nX 2>/dev/null | head -1 | tr -d ' ')
    case "$FINAL" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED)
        EC=$(sacct -j "$JID" --format=ExitCode -nX 2>/dev/null | head -1 | tr -d ' ')
        echo "DONE jobid=$JID state=${FINAL} exit=${EC:-?} at=$(date -Is)"
        echo "---- log tail ----"; tail -25 "$LOG" 2>/dev/null || echo "(no log at $LOG)"
        break ;;
      *)
        # sacct has NO terminal record on some clusters (podman jobs / allocations are not
        # accounted) -> the original code waited FOREVER here. Fall back: a job gone from
        # squeue whose LOG shows a completion marker, OR gone for K consecutive polls, is DONE.
        EMPTY=$((EMPTY+1))
        if tail -6 "$LOG" 2>/dev/null | grep -qE "\[exit=|\[smoke\] OK|no fingerprint saved|^DONE"; then
          echo "DONE jobid=$JID state=${FINAL:-gone} (log-marker) at=$(date -Is)"
          echo "---- log tail ----"; tail -25 "$LOG" 2>/dev/null
          break
        fi
        if [ "$EMPTY" -ge 2 ]; then
          echo "DONE jobid=$JID state=${FINAL:-gone} (squeue-empty x$EMPTY, sacct-unknown) at=$(date -Is)"
          echo "---- log tail ----"; tail -25 "$LOG" 2>/dev/null
          break
        fi
        echo "HEARTBEAT jobid=$JID state=(squeue-empty#$EMPTY, sacct=${FINAL:-unknown} -> confirming done)"
        sleep "$HB"; continue ;;
    esac
  fi
  EMPTY=0
  EL=$(( $(date +%s) - START ))
  LAST=$(tail -1 "$LOG" 2>/dev/null | cut -c1-80)
  echo "HEARTBEAT jobid=$JID state=$ST elapsed=${EL}s last='${LAST}'"
  sleep "$HB"
done
