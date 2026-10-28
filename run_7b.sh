#!/usr/bin/env bash
# Overnight tail: waits for the main queue (perturb + multiseed) to finish,
# then re-runs the 7B baseline with its OWN native chat template to close the
# prompt-fairness gap. Writes template_rerun_7b_summary.json.
set -u
cd "$(dirname "$0")"
PY=python

echo "=== 7B QUEUE START $(date '+%H:%M:%S') ==="
Q_PID="${1:-}"
if [ -n "$Q_PID" ]; then
  echo "[7b] waiting for queue PID $Q_PID to finish…"
  while kill -0 "$Q_PID" 2>/dev/null; do sleep 60; done
  echo "[7b] queue finished $(date '+%H:%M:%S')"
fi

echo "=== STAGE template_7b START $(date '+%H:%M:%S') ==="
"$PY" -u realdata/template_rerun.py --limit 480 --models 7b \
      --out template_rerun_7b_summary.json > run_template_7b.log 2>&1
echo "=== STAGE template_7b DONE rc=$? $(date '+%H:%M:%S') ==="
echo "=== 7B QUEUE ALL DONE $(date '+%H:%M:%S') ==="
