"""Agent-ablation study: quantify the contribution of the Auditor (Agent 3) and
the Arbiter gate (Agent 4) to PedSafe-Agent's safety behaviour.

Runs on the SYNTHETIC PediPharm benchmark, which (unlike FAERS) contains clean
fatal-contraindication traps (neonatal ceftriaxone+Ca, aspirin/Reye's, TMP-SMX
neonate, ...). FAERS cannot support this because it has no clean, labelled
contraindication context -- hence the synthetic benchmark for the safety claim.

Variants:
  full        : Extractor + Solver + Auditor + Arbiter gate   (the system)
  no_gate     : Auditor still runs, but the Arbiter does NOT block on a detected
                fatal contraindication  (isolates the gate's contribution)
  no_auditor  : Auditor removed entirely                       (isolates Agent 3)

Primary metric: fatal-contraindication BLOCK RECALL = of cases that truly carry
a fatal contraindication, the fraction the system blocks. Also reports the
false-block rate on safe cases (over-blocking).

Usage:  python realdata/ablation_study.py --per-category 50
"""
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from benchmark.generator import PediPharmBenchmarkGenerator
from agents.pipeline import PedSafePipeline


VARIANTS = {
    "full":       dict(run_auditor=True,  enforce_gate=True),
    "no_gate":    dict(run_auditor=True,  enforce_gate=False),
    "no_auditor": dict(run_auditor=False, enforce_gate=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=50)
    args = ap.parse_args()

    gen = PediPharmBenchmarkGenerator(seed=42)
    vigs = gen.generate_benchmark(count_per_category=args.per_category)
    n_fatal = sum(1 for v in vigs if v.ground_truth.has_fatal_contraindication)
    n_safe = len(vigs) - n_fatal
    print(f"[*] Synthetic benchmark: {len(vigs)} cases "
          f"({n_fatal} fatal-contraindication, {n_safe} safe).\n", flush=True)

    results = {}
    for name, flags in VARIANTS.items():
        print(f"[*] Running ablation variant: {name}  {flags}", flush=True)
        pipe = PedSafePipeline(**flags)
        tp = fn = fp = tn = 0  # fatal-block confusion vs ground-truth fatal
        t0 = time.time()
        for i, v in enumerate(vigs, 1):
            out = pipe.process(v.clinical_narrative)
            blocked = (out.final_status == "BLOCKED_FATAL")
            truly_fatal = v.ground_truth.has_fatal_contraindication
            if truly_fatal and blocked:      tp += 1
            elif truly_fatal and not blocked: fn += 1
            elif not truly_fatal and blocked: fp += 1
            else:                             tn += 1
            if i % max(1, len(vigs) // 10) == 0:
                print(f"    [{name}] {i}/{len(vigs)}  {time.time()-t0:.0f}s", flush=True)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        false_block = fp / (fp + tn) if (fp + tn) else 0.0
        results[name] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                         "fatal_block_recall": round(recall, 4),
                         "false_block_rate": round(false_block, 4)}
        print(f"    -> fatal-block recall={recall*100:.1f}%  "
              f"false-block rate={false_block*100:.1f}%\n", flush=True)

    print("=" * 78, flush=True)
    print(" ABLATION RESULTS (synthetic benchmark; fatal-contraindication safety)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Variant':<14} | {'Fatal-block recall':<20} | {'False-block rate':<18}", flush=True)
    print("-" * 78, flush=True)
    for name in VARIANTS:
        r = results[name]
        print(f"{name:<14} | {r['fatal_block_recall']*100:>6.1f}% "
              f"(tp={r['tp']},fn={r['fn']})   | {r['false_block_rate']*100:>6.1f}% "
              f"(fp={r['fp']},tn={r['tn']})", flush=True)
    print("=" * 78, flush=True)

    out_path = config.PROJECT_ROOT / "ablation_summary.json"
    json.dump({"n_cases": len(vigs), "n_fatal": n_fatal, "results": results},
              open(out_path, "w"), indent=2)
    print(f"\n[✓] Ablation summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
