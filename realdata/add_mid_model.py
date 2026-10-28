"""Add the mid-scale baseline (Phi-4-mini, 3.8B) to the completed real-FAERS
benchmark WITHOUT re-running the expensive 7B.

The cohort is deterministic (fixed seed, fixed limit) and every model is scored
at temperature 0, so running a new model over the identical 480 cases yields a
per-case correctness vector that aligns 1:1 with the vectors already saved in
eval_checkpoint_real.json (structure: saved[metric][system]). We therefore:
  1. run Phi-4-mini (and, with --models mid,3b, re-run the fast 3B as a
     determinism integrity check + to capture its raw outputs for the taxonomy)
     over the same cases,
  2. merge the new column with the saved PedSafe / 3B / 7B vectors,
  3. recompute the full paired significance suite as a 4-way comparison,
  4. dump per-case raw baseline outputs for the failure-taxonomy pass.

Crash-safe: the raw dump is appended per case and the per-case vectors are
checkpointed to a resume file, so an interruption never discards inference.
Re-invoking with the same --limit resumes from the last checkpointed case.

Usage:  python realdata/add_mid_model.py --limit 480 [--models mid | mid,3b]
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluate import MonolithicBaselineRunner, _load_vignettes, _score_case
from realdata import stats as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--models", default="mid",
                    help="comma list of extra models to run: mid[,3b]")
    args = ap.parse_args()
    run_names = [m.strip() for m in args.models.split(",") if m.strip()]
    run_3b = "3b" in run_names

    ckpt = json.load(open(config.PROJECT_ROOT / "eval_checkpoint_real.json"))
    saved = ckpt["vectors"]  # saved[metric][system]
    print(f"[*] Loaded saved vectors (cases_done={ckpt['cases_done']}): "
          f"dose n={len(saved['dose']['PedSafe-Agent'])}, "
          f"ceiling n={len(saved['ceiling']['PedSafe-Agent'])}", flush=True)

    vigs = _load_vignettes(True, None, 5, args.limit)

    # ---- resume support -----------------------------------------------------
    resume_path = config.PROJECT_ROOT / "add_mid_resume.json"
    dump_path = config.PROJECT_ROOT / "data" / "baseline_raw_dump.jsonl"
    systems = ["Monolithic-Mid"] + (["Monolithic-3B"] if run_3b else [])
    vec = {s: {"dose": [], "ceiling": []} for s in systems}
    start_idx = 0
    if resume_path.exists():
        r = json.load(open(resume_path))
        if r.get("limit") == args.limit and r.get("systems") == systems:
            vec = r["vec"]
            start_idx = r["cases_done"]
            print(f"[resume] continuing from case {start_idx}/{len(vigs)}", flush=True)

    mid = MonolithicBaselineRunner(model_path=config.DEFAULT_MID_BASELINE, name="Monolithic-Mid")
    runners = [("Monolithic-Mid", mid)]
    if run_3b:
        b3 = MonolithicBaselineRunner(model_path=config.DEFAULT_MONOLITHIC_BASELINE, name="Monolithic-3B")
        runners.append(("Monolithic-3B", b3))

    # fresh start truncates the raw dump; resume appends.
    dump_f = open(dump_path, "a" if start_idx else "w")

    t0 = time.time()
    for i, v in enumerate(vigs, 1):
        if i <= start_idx:
            continue
        gt, trap, narr = v.ground_truth, v.trap_category, v.clinical_narrative
        for name, runner in runners:
            out = runner.process(narr)
            sc = _score_case(gt, trap, out)
            if "dose" in sc:
                vec[name]["dose"].append(sc["dose"])
            if "ceiling" in sc:
                vec[name]["ceiling"].append(sc["ceiling"])
            dump_f.write(json.dumps({
                "case": i, "system": name, "drug": v.target_drug, "trap": trap,
                "label_min": gt.label_min_dose_mg, "label_max": gt.label_max_dose_mg,
                "adult_ceiling": gt.adult_ceiling_mg,
                "dose_eligible": bool(gt.dose_scoring_eligible and gt.label_min_dose_mg is not None),
                "parsed_dose": out["single_dose_mg"], "status": out["status"],
                "dose_ok": sc.get("dose"), "ceiling_ok": sc.get("ceiling"),
                "raw_output": out.get("raw_output", "")[:1200],
            }) + "\n")
        dump_f.flush()
        if i % max(1, len(vigs) // 20) == 0 or i == len(vigs):
            print(f"  [{i}/{len(vigs)}] elapsed {time.time()-t0:.0f}s", flush=True)
            json.dump({"limit": args.limit, "systems": systems,
                       "cases_done": i, "vec": vec}, open(resume_path, "w"))
    dump_f.close()

    # Integrity check (only meaningful if we re-ran 3B): fresh 3B must reproduce
    # the saved 3B vectors exactly (determinism at temperature 0).
    if run_3b:
        for m in ("dose", "ceiling"):
            a, bvec = vec["Monolithic-3B"][m], saved[m]["Monolithic-3B"]
            print(f"[integrity] 3B {m}: fresh n={len(a)} saved n={len(bvec)} "
                  f"identical={a == bvec}", flush=True)

    # Build the 4-way vectors (reuse saved PedSafe / 3B / 7B, add Mid).
    order = ["PedSafe-Agent", "Monolithic-3B", "Monolithic-Mid", "Monolithic-7B"]
    four = {}
    for m in ("dose", "ceiling"):
        four[m] = {
            "PedSafe-Agent": saved[m]["PedSafe-Agent"],
            "Monolithic-3B": saved[m]["Monolithic-3B"],
            "Monolithic-Mid": vec["Monolithic-Mid"][m],
            "Monolithic-7B": saved[m]["Monolithic-7B"],
        }

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print(" EMPIRICAL RESULTS (REAL FAERS DATA) -- 4-WAY", flush=True)
    print("=" * 100, flush=True)
    hdr = f"{'Metric':<30} | " + " | ".join(f"{s:<16}" for s in order)
    print(hdr, flush=True)
    print("-" * 100, flush=True)
    labels = {"dose": "Dose/Decision Accuracy (%)", "ceiling": "Adult Ceiling Adherence (%)"}
    for m in ("dose", "ceiling"):
        cells = []
        for s in order:
            vv = four[m][s]
            n = len(vv); ok = sum(vv)
            cells.append(f"{100*ok/n:4.1f}% (n={n})" if n else "n/a")
        print(f"{labels[m]:<30} | " + " | ".join(f"{c:<16}" for c in cells), flush=True)
    print("=" * 100, flush=True)

    # ---- significance -------------------------------------------------------
    print("\n SIGNIFICANCE TESTING (paired; reference = PedSafe-Agent)\n", flush=True)
    summary = {}
    pvals = {}
    for m in ("dose", "ceiling"):
        rep = S.summarize_metric(labels[m], four[m], reference="PedSafe-Agent")
        summary[m] = rep
        print(f"{labels[m]}:", flush=True)
        for s in order:
            info = rep["systems"][s]
            print(f"  {s:<16} rate={info['rate']*100:5.1f}%  "
                  f"wilson95=[{info['wilson_95ci'][0]*100:.1f}, {info['wilson_95ci'][1]*100:.1f}]  "
                  f"(n={info['n']})", flush=True)
        for s, mc in rep["mcnemar_vs_reference"].items():
            print(f"    McNemar vs {s:<16} b={mc['b']} c={mc['c']} p={mc['p_value']}", flush=True)
            pvals[f"mcnemar_{m}_PedSafe_vs_{s}"] = mc["p_value"]
        print("", flush=True)

    holm = S.holm_bonferroni(pvals)
    print("  Holm-Bonferroni across all comparisons:", flush=True)
    for tag, info in sorted(holm.items(), key=lambda kv: kv[1]["raw_p"]):
        star = "***" if info["holm_adj_p"] < 0.001 else ("*" if info["significant"] else "ns")
        print(f"    {tag:<46} raw_p={info['raw_p']:.5f} adj_p={info['holm_adj_p']:.5f} [{star}]", flush=True)
    print("=" * 100, flush=True)

    out_path = config.PROJECT_ROOT / "evaluation_summary_4way.json"
    json.dump({"order": order,
               "vectors_source": "merged: saved PedSafe/3B/7B + fresh Mid"
                                 + (" (+3B integrity re-run)" if run_3b else ""),
               "statistics": {"dose": summary["dose"], "ceiling": summary["ceiling"],
                              "holm_bonferroni": holm}},
              open(out_path, "w"), indent=2)
    print(f"\n[✓] 4-way summary -> {out_path}", flush=True)
    print(f"[✓] Raw outputs for taxonomy -> {dump_path}", flush=True)
    if resume_path.exists():
        resume_path.unlink()


if __name__ == "__main__":
    main()
