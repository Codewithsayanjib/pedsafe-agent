"""Perturbation robustness + extraction fidelity (caveat 3: "100% is too clean").

A flat 100% invites the suspicion that the system is tuned to pristine inputs.
This script stress-tests PedSafe-Agent two ways:

  1. EXTRACTION FIDELITY (clean data): for each dose-eligible case, compare the
     weight the Extractor (Agent 1) parses against the weight literally stated in
     the note (regex). This is the system's real fallible surface -- the symbolic
     solver is exact by construction, so any end-to-end error originates here.
     The reported fidelity is < 100%, which is the honest counterpart to the
     100% decision accuracy.

  2. GRACEFUL DEGRADATION: re-run the full pipeline on the SAME notes after
     meaning-preserving noise -- typos, random casing, sentence reordering, and
     an injected distractor sentence. Numeric values are never altered (that
     would change ground truth). A robust system degrades gently rather than
     staying suspiciously perfect or collapsing.

Usage:  python realdata/perturbation_robustness.py --sample 60
"""
import re
import sys
import json
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluate import _load_vignettes, _score_case, _normalize_pedsafe
from agents.pipeline import PedSafePipeline

DISTRACTOR = ("The ward was busy overnight and the parents asked several "
              "questions about discharge planning.")


def typo(text, rng, rate=0.04):
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap" and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "dup":
                chars[i] = chars[i] * 2
            elif op == "drop":
                chars[i] = ""
    return "".join(chars)


def random_case(text, rng):
    return "".join(c.upper() if rng.random() < 0.3 else c.lower() for c in text)


def reorder(text, rng):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    rng.shuffle(parts)
    return " ".join(parts)


def add_distractor(text, rng):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    pos = rng.randint(0, len(parts))
    parts.insert(pos, DISTRACTOR)
    return " ".join(parts)


PERTURBATIONS = {
    "clean": lambda t, rng: t,
    "typos": typo,
    "random_case": random_case,
    "reorder": reorder,
    "distractor": add_distractor,
}


def stated_weight_kg(note):
    """The weight literally written in the note, in kg (regex, for fidelity)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg\b", note, re.IGNORECASE)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    args = ap.parse_args()

    pool = _load_vignettes(True, None, 5, 480)
    eligible = [v for v in pool
                if v.ground_truth.dose_scoring_eligible
                and v.ground_truth.label_min_dose_mg is not None]
    rng = random.Random(42)
    rng.shuffle(eligible)
    sample = eligible[:args.sample]
    print(f"[*] Perturbation robustness on {len(sample)} dose-eligible cases.\n", flush=True)

    pipe = PedSafePipeline()

    # ---- 1. extraction fidelity on clean notes ------------------------------
    fid_match = fid_total = 0
    for v in sample:
        stated = stated_weight_kg(v.clinical_narrative)
        if stated is None:
            continue
        prof = pipe.extractor.extract(v.clinical_narrative)
        fid_total += 1
        if abs(prof.weight_kg - stated) <= max(0.1, 0.05 * stated):
            fid_match += 1
    fidelity = fid_match / fid_total if fid_total else 0.0
    print(f"[extraction fidelity] weight parsed within 5% of the stated value in "
          f"{fid_match}/{fid_total} notes = {fidelity*100:.1f}%\n", flush=True)

    # ---- 2. graceful degradation across perturbations -----------------------
    results = {}
    for pname, pfn in PERTURBATIONS.items():
        correct = total = 0
        t0 = time.time()
        prng = random.Random(100)
        for v in sample:
            note = pfn(v.clinical_narrative, prng)
            out = _normalize_pedsafe(pipe.process(note))
            sc = _score_case(v.ground_truth, v.trap_category, out)
            if "dose" in sc:
                total += 1
                correct += sc["dose"]
        rate = correct / total if total else 0.0
        results[pname] = {"dose_rate": round(rate, 4), "n": total}
        print(f"  [{pname:<12}] dose accuracy {rate*100:5.1f}%  (n={total})  "
              f"{time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" PERTURBATION ROBUSTNESS (PedSafe-Agent dose accuracy)", flush=True)
    print("=" * 70, flush=True)
    clean = results["clean"]["dose_rate"]
    for pname, r in results.items():
        delta = (r["dose_rate"] - clean) * 100
        tag = "(baseline)" if pname == "clean" else f"(Δ {delta:+.1f} pts)"
        print(f"  {pname:<12} {r['dose_rate']*100:5.1f}%  {tag}", flush=True)
    print("=" * 70, flush=True)

    out = config.PROJECT_ROOT / "perturbation_summary.json"
    json.dump({"sample": len(sample),
               "extraction_fidelity": {"match": fid_match, "total": fid_total,
                                        "rate": round(fidelity, 4)},
               "degradation": results}, open(out, "w"), indent=2)
    print(f"\n[✓] Perturbation summary -> {out}", flush=True)


if __name__ == "__main__":
    main()
