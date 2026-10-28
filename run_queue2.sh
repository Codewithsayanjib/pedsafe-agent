#!/usr/bin/env bash
# Corrected queue tail: waits for the running template stage, then runs
# perturbation and a 2-seed multi-seed eval. Replaces the original orchestrator.
set -u
cd "$(dirname "$0")"
PY=python

echo "=== QUEUE2 START $(date '+%H:%M:%S') ==="
TPL_PID="${1:-}"
if [ -n "$TPL_PID" ]; then
  echo "[queue2] waiting for template PID $TPL_PID to finish…"
  while kill -0 "$TPL_PID" 2>/dev/null; do sleep 30; done
  echo "[queue2] template finished $(date '+%H:%M:%S')"
fi

run_stage () {
  local name="$1"; shift
  local log="$1"; shift
  echo "=== STAGE $name START $(date '+%H:%M:%S') ==="
  "$PY" -u "$@" > "$log" 2>&1
  echo "=== STAGE $name DONE rc=$? $(date '+%H:%M:%S') ==="
}

run_stage perturb     run_perturb.log      realdata/perturbation_robustness.py --sample 60
run_stage multiseed   run_multiseed.log    realdata/multiseed_eval.py --limit 480 --seeds 7,13

echo "=== QUEUE2 ALL DONE $(date '+%H:%M:%S') ==="
