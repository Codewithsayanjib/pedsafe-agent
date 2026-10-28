"""Audit WHY the monolithic baselines fail dose accuracy.

Re-runs PedSafe + 3B + 7B on the first N dose-eligible FAERS cases and prints,
for each, the ground-truth label-permitted dose window, every system's parsed
dose + pass/fail, and the baselines' RAW model output. This confirms whether the
~2-5% baseline accuracy is a genuine reasoning failure (wrong numbers) rather
than a scoring/parsing artifact -- the key validity question for reviewers.

Usage:  python realdata/audit_baseline_failures.py --n 8
"""
import sys
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from benchmark.schema import SyntheticVignette
from agents.pipeline import PedSafePipeline
from evaluate import MonolithicBaselineRunner, _normalize_pedsafe, _score_case


def _clean(raw: str) -> str:
    """Strip DeepSeek <think> blocks and fences for readable display."""
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="dose-eligible cases to audit")
    args = ap.parse_args()

    path = config.BENCHMARK_DIR / "faers_cohort.json"
    vigs = [SyntheticVignette(**d) for d in json.loads(path.read_text())]
    eligible = [v for v in vigs
                if v.ground_truth.dose_scoring_eligible
                and v.ground_truth.label_min_dose_mg is not None
                and not v.ground_truth.has_fatal_contraindication][:args.n]

    print(f"[*] Auditing {len(eligible)} dose-eligible cases (of {len(vigs)} total).\n", flush=True)

    pedsafe = PedSafePipeline()
    b3 = MonolithicBaselineRunner(model_path=config.DEFAULT_MONOLITHIC_BASELINE, name="Monolithic-3B")
    b7 = MonolithicBaselineRunner(model_path=config.DEFAULT_7B_BASELINE, name="Monolithic-7B")

    for i, v in enumerate(eligible, 1):
        gt, trap, narr = v.ground_truth, v.trap_category, v.clinical_narrative
        lo, hi = gt.label_min_dose_mg, gt.label_max_dose_mg

        ps = _normalize_pedsafe(pedsafe.process(narr))
        o3 = b3.process(narr)
        o7 = b7.process(narr)

        s_ps = _score_case(gt, trap, ps).get("dose")
        s_3 = _score_case(gt, trap, o3).get("dose")
        s_7 = _score_case(gt, trap, o7).get("dose")

        print("=" * 90, flush=True)
        print(f"CASE {i}  [{v.target_drug}]  trap={trap}", flush=True)
        print(f"  Narrative: {narr[:260]}", flush=True)
        print(f"  GROUND-TRUTH label-permitted single-dose window: {lo:.1f}-{hi:.1f} mg "
              f"(adult ceiling {gt.adult_ceiling_mg})", flush=True)
        print(f"  PedSafe-Agent : dose={ps['single_dose_mg']:.1f} mg  -> {'PASS' if s_ps else 'FAIL'}", flush=True)
        print(f"  Monolithic-3B : parsed dose={o3['single_dose_mg']:.1f} mg  -> {'PASS' if s_3 else 'FAIL'}", flush=True)
        print(f"     raw-3B: {_clean(o3['raw_output'])[:400]}", flush=True)
        print(f"  Monolithic-7B : parsed dose={o7['single_dose_mg']:.1f} mg  -> {'PASS' if s_7 else 'FAIL'}", flush=True)
        print(f"     raw-7B: {_clean(o7['raw_output'])[:400]}", flush=True)
        print("", flush=True)

    print("[*] Audit complete.", flush=True)


if __name__ == "__main__":
    main()
