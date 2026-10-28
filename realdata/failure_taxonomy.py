"""Classify WHY each monolithic baseline fails a dose-eligible case, from the
raw per-case outputs captured in data/baseline_raw_dump.jsonl.

Failure buckets (dose-eligible, non-fatal cases only):
  over_block   : refused to dose (status == BLOCKED_FATAL) a case where dosing
                 was clinically appropriate -- clinically useless.
  overdose     : computed dose above the label-permitted ceiling -- dangerous.
  underdose    : computed a positive dose below the label window (typically a
                 mg/kg-vs-total unit confusion or a fixed unit dose).
  no_dose      : emitted no usable dose (parsed 0 mg).
  correct      : dose landed in the permitted window.

Usage:  python realdata/failure_taxonomy.py
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def classify(r):
    if not r["dose_eligible"]:
        return None  # not scored for dose
    if r.get("dose_ok") == 1:
        return "correct"
    if r["status"] == "BLOCKED_FATAL":
        return "over_block"
    d = r["parsed_dose"]
    if d <= 0:
        return "no_dose"
    if r["label_max"] is not None and d > r["label_max"] * 1.02:
        return "overdose"
    if r["label_min"] is not None and d < r["label_min"] * 0.98:
        return "underdose"
    return "other"


def main():
    path = config.PROJECT_ROOT / "data" / "baseline_raw_dump.jsonl"
    rows = [json.loads(l) for l in open(path)]
    buckets = ["correct", "over_block", "overdose", "underdose", "no_dose", "other"]
    tally = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c = classify(r)
        if c is None:
            continue
        tally[r["system"]][c] += 1

    print("=" * 84, flush=True)
    print(" BASELINE FAILURE TAXONOMY (dose-eligible cases)", flush=True)
    print("=" * 84, flush=True)
    print(f"{'System':<16} | " + " | ".join(f"{b:<10}" for b in buckets) + " | total", flush=True)
    print("-" * 84, flush=True)
    for sysname, counts in tally.items():
        total = sum(counts.values())
        cells = []
        for b in buckets:
            n = counts.get(b, 0)
            cells.append(f"{n:>3} {100*n/total:4.0f}%" if total else "  0")
        print(f"{sysname:<16} | " + " | ".join(f"{c:<10}" for c in cells) + f" | {total}", flush=True)
    print("=" * 84, flush=True)
    json.dump({s: dict(c) for s, c in tally.items()},
              open(config.PROJECT_ROOT / "failure_taxonomy.json", "w"), indent=2)
    print(f"\n[✓] Taxonomy -> {config.PROJECT_ROOT / 'failure_taxonomy.json'}", flush=True)


if __name__ == "__main__":
    main()
