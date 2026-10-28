#!/usr/bin/env bash
# Self-chaining GPU job queue for the remaining experiments. Waits for the
# running ablation to finish, then runs each stage sequentially (no GPU
# contention), logging each to its own file and printing a DONE marker.
set -u
cd "$(dirname "$0")"
PY=python

echo "=== QUEUE START $(date '+%H:%M:%S') ==="

# 1) wait for the ablation process to exit (if still running)
ABL_PID="${1:-}"
if [ -n "$ABL_PID" ]; then
  echo "[queue] waiting for ablation PID $ABL_PID to finish…"
  while kill -0 "$ABL_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] ablation finished $(date '+%H:%M:%S')"
fi

run_stage () {
  local name="$1"; shift
  local log="$1"; shift
  echo "=== STAGE $name START $(date '+%H:%M:%S') ==="
  "$PY" -u "$@" > "$log" 2>&1
  echo "=== STAGE $name DONE rc=$? $(date '+%H:%M:%S') ==="
}

run_stage latency     run_latency.log      realdata/latency_benchmark.py --sample 25
run_stage template    run_template.log     realdata/template_rerun.py --limit 480
run_stage perturb     run_perturb.log      realdata/perturbation_robustness.py --sample 60
run_stage multiseed   run_multiseed.log    realdata/multiseed_eval.py --limit 480 --seeds 7,13,21

echo "=== QUEUE ALL DONE $(date '+%H:%M:%S') ==="
