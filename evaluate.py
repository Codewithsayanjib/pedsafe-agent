import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from benchmark.generator import PediPharmBenchmarkGenerator
from benchmark.schema import SyntheticVignette
from agents.pipeline import PedSafePipeline
from agents.base import MLXModelManager


class MonolithicBaselineRunner:
    """
    Evaluates standalone monolithic LLM/SLM without multi-agent decomposition
    or symbolic tools (standard prompt-to-prescription baseline).
    """
    def __init__(self, model_path: str = config.DEFAULT_MONOLITHIC_BASELINE, name: str = "Monolithic-3B"):
        self.model_path = model_path
        self.name = name
        self.manager = MLXModelManager()
        # Reasoning-distilled models (e.g. DeepSeek-R1) emit a long <think> block
        # before the answer; 512 tokens truncates them before they ever produce
        # the final JSON, which is unfair and inflates the baseline failure rate.
        # Give them a budget large enough to conclude.
        low = model_path.lower()
        self.max_tokens = 2048 if ("r1" in low or "deepseek" in low) else 512

    # System + user content shared by every baseline prompt variant.
    _SYSTEM_PROMPT = (
        "You are an expert pediatric clinical dosing assistant. Read the patient "
        "referral note and calculate the exact single dose in mg, dosing interval "
        "in hours, and determine whether the prescription should be APPROVED or "
        "BLOCKED due to contraindications.\n\nRespond in this exact JSON format:\n"
        "{\n  \"single_dose_mg\": float,\n  \"interval_hours\": integer,\n"
        "  \"is_adult_ceiling_capped\": boolean,\n"
        "  \"status\": \"APPROVED\" or \"BLOCKED_FATAL\",\n  \"reasoning\": string\n}"
    )

    def _build_prompt(self, clinical_text: str) -> str:
        """Default baseline prompt: hard-coded Llama-3 chat tokens (kept for
        continuity with the completed runs). Subclasses override this to use a
        model's own native chat template."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self._SYSTEM_PROMPT}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Patient Referral:\n{clinical_text}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def process(self, clinical_text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(clinical_text)
        start = time.time()
        # Allocate sufficient tokens for reasoning models (e.g. DeepSeek-R1)
        res = self.manager.run_inference(self.model_path, prompt, max_tokens=self.max_tokens, temperature=0.0)
        elapsed = time.time() - start
        
        raw_text = res["text"]
        # Strip DeepSeek-R1 thinking tokens
        cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Parse output
        single_dose = 0.0
        interval = 8
        is_capped = False
        status = "APPROVED"

        try:
            cleaned = re.sub(r"```json\s*", "", cleaned_text)
            cleaned = re.sub(r"```", "", cleaned).strip()
            json_m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_m:
                cleaned = json_m.group(0)
            data = json.loads(cleaned)
            single_dose = float(data.get("single_dose_mg", 0.0))
            interval = int(data.get("interval_hours", 8))
            is_capped = bool(data.get("is_adult_ceiling_capped", False))
            status = str(data.get("status", "APPROVED")).upper()
        except Exception:
            # Fallback when JSON is absent/malformed (e.g. a reasoning model that
            # ran out of tokens mid-answer). Take the LAST mg figure in the text,
            # which is the model's final computed dose, not an intermediate
            # per-kg rate quoted early in its reasoning (comma-grouped allowed).
            nums = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*mg\b", cleaned_text, re.IGNORECASE)
            if nums:
                single_dose = float(nums[-1].replace(",", ""))
            if "block" in cleaned_text.lower() or "contraindicat" in cleaned_text.lower():
                status = "BLOCKED_FATAL"

        return {
            "single_dose_mg": single_dose,
            "interval_hours": interval,
            "is_adult_ceiling_capped": is_capped,
            "status": status,
            "execution_time_seconds": round(elapsed, 3),
            "raw_output": raw_text
        }


def _load_vignettes(real, cohort_path, num_samples, limit):
    """Return list of SyntheticVignette, either real FAERS cohort or synthetic."""
    if real:
        import json
        path = Path(cohort_path) if cohort_path else (config.BENCHMARK_DIR / "faers_cohort.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Real cohort not found at {path}. Build it first:\n"
                f"  python realdata/build_cohort.py --per-drug 400")
        data = json.loads(path.read_text())
        vigs = [SyntheticVignette(**d) for d in data]
        print(f"[*] Loaded {len(vigs)} REAL pediatric cases from openFDA FAERS ({path.name}).")
    else:
        gen = PediPharmBenchmarkGenerator(seed=42)
        path = config.BENCHMARK_DIR / f"pedipharm_eval_{num_samples * 5}.json"
        print(f"[*] Generating {num_samples} vignettes per trap category (Total: {num_samples * 5})...")
        vigs = gen.generate_benchmark(count_per_category=num_samples)
        gen.save_to_file(vigs, path)
    if limit:
        vigs = vigs[:limit]
        print(f"[*] Limited to first {len(vigs)} cases for this run.")
    return vigs


def _normalize_pedsafe(ps_out):
    """Project the PedSafe pipeline output onto the common scoring shape."""
    return {
        "single_dose_mg": ps_out.recommended_single_dose_mg,
        "interval_hours": ps_out.interval_hours,
        "is_adult_ceiling_capped": ps_out.is_adult_ceiling_capped,
        "status": ps_out.final_status,
        "execution_time_seconds": ps_out.execution_time_seconds,
    }


def _score_case(gt, trap_category, out):
    """Return {metric: 0/1} for every metric that APPLIES to this case.

    Dose scoring is RANGE-based (not exact match), which is non-circular and
    clinically defensible: a recommendation is correct if it lands within the
    label-permitted mg/kg window for the patient's real weight, with the adult
    ceiling applied. Ceiling adherence asks only that the dose never exceeds the
    adult maximum. A small relative tolerance absorbs rounding."""
    scores = {}
    dose = out["single_dose_mg"]

    if gt.has_fatal_contraindication:
        blocked = out["status"] == "BLOCKED_FATAL"
        scores["fatal"] = int(blocked)
        return scores  # a blocked case has no dose to score

    # Dose-accuracy (range) -- only for drugs with a well-defined mg/kg range.
    if gt.dose_scoring_eligible and gt.label_min_dose_mg is not None:
        lo = gt.label_min_dose_mg * 0.98
        hi = gt.label_max_dose_mg * 1.02
        in_range = (dose >= lo) and (dose <= hi) and dose > 0
        scores["dose"] = int(in_range)

    # Ceiling adherence -- dose must not exceed the adult maximum (never overdose).
    if trap_category == "obesity_ceiling" and gt.adult_ceiling_mg is not None:
        scores["ceiling"] = int(0 < dose <= gt.adult_ceiling_mg * 1.02)

    if gt.is_renal_adjusted:
        scores["renal"] = int(out["interval_hours"] == gt.recommended_interval_hours)
    return scores


def run_evaluation(num_samples=5, run_3b=True, run_7b=True,
                   real=False, cohort_path=None, limit=None):
    print("=" * 85)
    print(" PedSafe-Agent Empirical Evaluation Benchmark ")
    print(" Multi-Agent SLM vs Monolithic 3B and 7B Baselines on Pediatric Pharmacotherapy")
    print(f" Dataset: {'REAL openFDA FAERS (CC0)' if real else 'synthetic PediPharm'}")
    print("=" * 85)

    vignettes = _load_vignettes(real, cohort_path, num_samples, limit)

    print("[*] Initializing PedSafe-Agent Multi-Agent SLM Pipeline...")
    pedsafe = PedSafePipeline()
    baseline_3b = baseline_7b = None
    if run_3b:
        print(f"[*] Initializing Monolithic 3B Baseline ({config.DEFAULT_MONOLITHIC_BASELINE})...")
        baseline_3b = MonolithicBaselineRunner(model_path=config.DEFAULT_MONOLITHIC_BASELINE, name="Llama-3.2-3B")
    if run_7b:
        print(f"[*] Initializing Monolithic 7B Baseline ({config.DEFAULT_7B_BASELINE})...")
        baseline_7b = MonolithicBaselineRunner(model_path=config.DEFAULT_7B_BASELINE, name="DeepSeek-R1-7B")

    systems = ["PedSafe-Agent"] + (["Monolithic-3B"] if run_3b else []) + (["Monolithic-7B"] if run_7b else [])

    # Aggregate counts (for the headline table) + per-case vectors (for stats).
    METRICS = ["dose", "ceiling", "fatal", "renal"]
    counts = {s: {m + "_ok": 0 for m in METRICS} | {m + "_n": 0 for m in METRICS} | {"time": 0.0} for s in systems}
    vectors = {m: {s: [] for s in systems} for m in METRICS}

    print("\n[*] Commencing evaluation across test cases...\n")
    runners = {"PedSafe-Agent": lambda t: _normalize_pedsafe(pedsafe.process(t))}
    if run_3b:
        runners["Monolithic-3B"] = lambda t: baseline_3b.process(t)
    if run_7b:
        runners["Monolithic-7B"] = lambda t: baseline_7b.process(t)

    import time as _time
    ckpt_path = config.PROJECT_ROOT / ("eval_checkpoint_real.json" if real else "eval_checkpoint.json")
    for i, v in enumerate(vignettes, 1):
        gt, trap, narrative = v.ground_truth, v.trap_category, v.clinical_narrative
        _t0 = _time.time()
        per_case_lat = {}
        for s in systems:
            _s0 = _time.time()
            out = runners[s](narrative)
            per_case_lat[s] = _time.time() - _s0
            counts[s]["time"] += out.get("execution_time_seconds", 0.0)
            sc = _score_case(gt, trap, out)
            for m, ok in sc.items():
                counts[s][m + "_ok"] += ok
                counts[s][m + "_n"] += 1
                vectors[m][s].append(ok)
        # Per-case progress, flushed immediately so it survives to the log even
        # when stdout is piped (block-buffered otherwise).
        lat_str = " ".join(f"{s}={per_case_lat[s]:.1f}s" for s in systems)
        print(f"  [{i}/{len(vignettes)}] {_time.time()-_t0:.1f}s ({lat_str})", flush=True)
        # Checkpoint partial results every 10 cases so a teardown is never a
        # total loss: whatever completed can still be scored.
        if i % 10 == 0 or i == len(vignettes):
            try:
                with open(ckpt_path, "w") as _cf:
                    json.dump({"cases_done": i, "counts": counts, "vectors": vectors}, _cf)
            except Exception:
                pass

    n = len(vignettes)

    # ---- headline comparative table -----------------------------------------
    print("\n" + "=" * 95)
    print(" EMPIRICAL RESULTS " + ("(REAL FAERS DATA)" if real else "(SYNTHETIC)"))
    print("=" * 95)
    print(f"{'Metric':<34} | " + " | ".join(f"{s:<16}" for s in systems))
    print("-" * 95)
    labels = {"dose": "Dose/Decision Accuracy (%)", "ceiling": "Adult Ceiling Adherence (%)",
              "fatal": "Fatal Contraindication Recall (%)", "renal": "Renal Adjustment Rate (%)"}
    for m in METRICS:
        if sum(counts[s][m + "_n"] for s in systems) == 0:
            continue
        row = f"{labels[m]:<34} | "
        cells = []
        for s in systems:
            d = counts[s][m + "_n"]
            cells.append(f"{(counts[s][m+'_ok']/d*100 if d else 0):.1f}% (n={d})".ljust(16))
        print(row + " | ".join(cells))
    lat = f"{'Avg Latency (s/case)':<34} | " + " | ".join(f"{counts[s]['time']/n:.2f}s".ljust(16) for s in systems)
    print(lat)
    print("=" * 95)

    # ---- significance testing -----------------------------------------------
    stats_report = {}
    try:
        from realdata import stats as st
        print("\n" + "=" * 95)
        print(" SIGNIFICANCE TESTING (paired; reference = PedSafe-Agent)")
        print("=" * 95)
        all_pvals = {}
        for m in METRICS:
            if not any(vectors[m][s] for s in systems):
                continue
            rep = st.summarize_metric(labels[m], vectors[m], reference="PedSafe-Agent")
            stats_report[m] = rep
            print(f"\n{labels[m]}:")
            for s in systems:
                info = rep["systems"][s]
                print(f"  {s:<16} rate={info['rate']*100:5.1f}%  "
                      f"bootstrap95=[{info['bootstrap_95ci'][0]*100:.1f}, {info['bootstrap_95ci'][1]*100:.1f}]  (n={info['n']})")
            for s, mc in rep["mcnemar_vs_reference"].items():
                tag = f"mcnemar_{m}_PedSafe_vs_{s}"
                all_pvals[tag] = mc["p_value"]
                print(f"    McNemar vs {s:<14} b={mc['b']} c={mc['c']}  p={mc['p_value']}")
        if all_pvals:
            print("\n  Holm-Bonferroni correction across all comparisons:")
            corrected = st.holm_bonferroni(all_pvals)
            for tag, info in corrected.items():
                star = "***" if info["significant"] else "ns"
                print(f"    {tag:<48} raw_p={info['raw_p']:.5f}  adj_p={info['holm_adj_p']:.5f}  [{star}]")
            stats_report["holm_bonferroni"] = corrected
        print("=" * 95)
    except Exception as e:  # noqa: BLE001
        print(f"[!] Stats module skipped: {e}")

    # ---- persist -------------------------------------------------------------
    summary_path = config.PROJECT_ROOT / ("evaluation_summary_real.json" if real else "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"dataset": "real_faers" if real else "synthetic",
                   "total_test_cases": n, "systems": systems,
                   "counts": counts, "statistics": stats_report}, f, indent=2)
    print(f"\n[✓] Metrics saved to: {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PedSafe-Agent vs Baselines")
    parser.add_argument("--count", type=int, default=2, help="Synthetic cases per category (total = count*5)")
    parser.add_argument("--real", action="store_true", help="Use the real openFDA FAERS cohort")
    parser.add_argument("--cohort", type=str, default=None, help="Path to real cohort JSON")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of cases (for smoke tests)")
    parser.add_argument("--skip-3b", action="store_true", help="Skip 3B baseline")
    parser.add_argument("--skip-7b", action="store_true", help="Skip 7B baseline")
    args = parser.parse_args()
    run_evaluation(num_samples=args.count, run_3b=not args.skip_3b, run_7b=not args.skip_7b,
                   real=args.real, cohort_path=args.cohort, limit=args.limit)
