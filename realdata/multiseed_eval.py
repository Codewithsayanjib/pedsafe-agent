"""Multi-seed robustness eval (caveat 4: single-run result).

The headline numbers come from one fixed 480-case slice (the first 480 of the
2202-case FAERS pool). To show the result is not a single-sample fluke, we
re-draw the 480-case evaluation set under several additional random seeds
(shuffling the full pool), re-run PedSafe-Agent + Monolithic-3B + Monolithic-Mid
on each, and report mean +/- SD of dose accuracy and ceiling adherence across
seeds. The 7B is excluded (8 h/seed); its single-seed number stands with a note.

Each seed is checkpointed, so the run resumes after an interruption.

Usage:  python realdata/multiseed_eval.py --limit 480 --seeds 7,13,21
"""
import sys
import json
import time
import random
import argparse
import statistics as stx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluate import (MonolithicBaselineRunner, _load_vignettes, _score_case,
                      _normalize_pedsafe)
from agents.pipeline import PedSafePipeline


def _rate(vec):
    return (sum(vec) / len(vec)) if vec else 0.0


def run_seed(seed, pool, limit, pipe, mono3, monomid):
    """Score the three systems on a seed-shuffled 480-case draw."""
    idx = list(range(len(pool)))
    random.Random(seed).shuffle(idx)
    vigs = [pool[i] for i in idx[:limit]]
    systems = {"PedSafe-Agent": lambda t: _normalize_pedsafe(pipe.process(t)),
               "Monolithic-3B": lambda t: mono3.process(t),
               "Monolithic-Mid": lambda t: monomid.process(t)}
    vec = {s: {"dose": [], "ceiling": []} for s in systems}
    t0 = time.time()
    for i, v in enumerate(vigs, 1):
        for s, fn in systems.items():
            sc = _score_case(v.ground_truth, v.trap_category, fn(v.clinical_narrative))
            if "dose" in sc:
                vec[s]["dose"].append(sc["dose"])
            if "ceiling" in sc:
                vec[s]["ceiling"].append(sc["ceiling"])
        if i % max(1, len(vigs) // 10) == 0 or i == len(vigs):
            print(f"    [seed {seed}] {i}/{len(vigs)} elapsed {time.time()-t0:.0f}s", flush=True)
    return {s: {"dose_rate": round(_rate(vec[s]["dose"]), 4),
                "dose_n": len(vec[s]["dose"]),
                "ceiling_rate": round(_rate(vec[s]["ceiling"]), 4),
                "ceiling_n": len(vec[s]["ceiling"])} for s in systems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--seeds", default="7,13,21")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    pool = _load_vignettes(True, None, 5, None)  # full 2202-case pool
    print(f"[*] Pool size {len(pool)}; seeds {seeds}; limit {args.limit}/seed\n", flush=True)

    ckpt_path = config.PROJECT_ROOT / "multiseed_checkpoint.json"
    done = {}
    if ckpt_path.exists():
        saved = json.load(open(ckpt_path))
        if saved.get("limit") == args.limit:
            done = {int(k): v for k, v in saved.get("per_seed", {}).items()}
            print(f"[resume] seeds already done: {sorted(done)}", flush=True)

    pipe = PedSafePipeline()
    mono3 = MonolithicBaselineRunner(model_path=config.DEFAULT_MONOLITHIC_BASELINE, name="Monolithic-3B")
    monomid = MonolithicBaselineRunner(model_path=config.DEFAULT_MID_BASELINE, name="Monolithic-Mid")

    for sd in seeds:
        if sd in done:
            continue
        print(f"[*] Seed {sd} …", flush=True)
        done[sd] = run_seed(sd, pool, args.limit, pipe, mono3, monomid)
        json.dump({"limit": args.limit, "per_seed": {str(k): v for k, v in done.items()}},
                  open(ckpt_path, "w"), indent=2)

    # ---- aggregate (include the original fixed slice as "seed 42/order") ----
    try:
        four = json.load(open(config.PROJECT_ROOT / "evaluation_summary_4way.json"))
        orig = {s: {"dose_rate": four["statistics"]["dose"]["systems"][s]["rate"],
                    "ceiling_rate": four["statistics"]["ceiling"]["systems"][s]["rate"]}
                for s in ("PedSafe-Agent", "Monolithic-3B", "Monolithic-Mid")}
    except Exception:
        orig = None

    sysnames = ["PedSafe-Agent", "Monolithic-3B", "Monolithic-Mid"]
    print("\n" + "=" * 92, flush=True)
    print(" MULTI-SEED ROBUSTNESS  (dose accuracy across seeds; mean +/- SD)", flush=True)
    print("=" * 92, flush=True)
    agg = {}
    for s in sysnames:
        rates = [done[sd][s]["dose_rate"] for sd in seeds]
        if orig:
            rates = rates + [orig[s]["dose_rate"]]
        m = stx.mean(rates); sd_ = stx.pstdev(rates) if len(rates) > 1 else 0.0
        agg[s] = {"dose_rates": rates, "mean": round(m, 4), "sd": round(sd_, 4)}
        seedstr = "  ".join(f"{r*100:.1f}" for r in rates)
        print(f"  {s:<16} dose%: [{seedstr}]  ->  mean {m*100:.1f}% +/- {sd_*100:.1f}", flush=True)
    print("=" * 92, flush=True)

    out = config.PROJECT_ROOT / "multiseed_summary.json"
    json.dump({"seeds": seeds, "limit": args.limit,
               "includes_original_slice": bool(orig),
               "per_seed": {str(k): v for k, v in done.items()},
               "aggregate_dose": agg}, open(out, "w"), indent=2)
    print(f"\n[✓] Multi-seed summary -> {out}", flush=True)
    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
