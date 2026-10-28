"""Native chat-template fairness re-run (caveat 2: prompt fairness).

The completed benchmark fed every monolithic baseline the same hard-coded
Llama-3 chat tokens. A reviewer could argue that disadvantages non-Llama models
(e.g. Phi-4-mini) whose native chat format differs. To remove that asymmetry we
re-score the two cheap baselines (Monolithic-3B = Llama-3.2-3B, Monolithic-Mid =
Phi-4-mini) using EACH model's OWN chat template via tokenizer.apply_chat_template,
holding the system+user content identical, and compare to the original numbers.

If the rates barely move, the gap to PedSafe is not a prompt artifact. (The 7B
is excluded: an 8 h re-run; its single-prompt number stays with a stated note.)

Usage:  python realdata/template_rerun.py --limit 480
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


class TemplatedBaselineRunner(MonolithicBaselineRunner):
    """Baseline that builds its prompt from the model's own chat template."""
    def _build_prompt(self, clinical_text: str) -> str:
        _, tokenizer = self.manager.get_model(self.model_path)
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient Referral:\n{clinical_text}"},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Some templates reject a system role; fold it into the user turn.
            merged = [{"role": "user",
                       "content": self._SYSTEM_PROMPT + "\n\nPatient Referral:\n" + clinical_text}]
            return tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True)


def _rate(vec):
    return (sum(vec) / len(vec)) if vec else 0.0


MODEL_MAP = {
    "3b":  ("Monolithic-3B", config.DEFAULT_MONOLITHIC_BASELINE),
    "mid": ("Monolithic-Mid", config.DEFAULT_MID_BASELINE),
    "7b":  ("Monolithic-7B", config.DEFAULT_7B_BASELINE),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--models", default="3b,mid",
                    help="comma list from {3b,mid,7b}")
    ap.add_argument("--out", default="template_rerun_summary.json")
    args = ap.parse_args()
    targets = [MODEL_MAP[m.strip()] for m in args.models.split(",") if m.strip()]

    # Original (hard-coded-prompt) numbers to compare against. Per-case vectors
    # exist for 3B and 7B (checkpoint) -> paired McNemar; Mid has rate only.
    ckpt = json.load(open(config.PROJECT_ROOT / "eval_checkpoint_real.json"))["vectors"]
    four = json.load(open(config.PROJECT_ROOT / "evaluation_summary_4way.json"))
    orig = {n: {"dose": ckpt["dose"][n], "ceiling": ckpt["ceiling"][n]}
            for n in ("Monolithic-3B", "Monolithic-7B") if n in ckpt["dose"]}

    vigs = _load_vignettes(True, None, 5, args.limit)

    report = {}
    for name, mpath in targets:
        runner = TemplatedBaselineRunner(model_path=mpath, name=name)
        vec = {"dose": [], "ceiling": []}
        t0 = time.time()
        print(f"[*] {name}: native-template re-run over {len(vigs)} cases …", flush=True)
        for i, v in enumerate(vigs, 1):
            out = runner.process(v.clinical_narrative)
            sc = _score_case(v.ground_truth, v.trap_category, out)
            if "dose" in sc:
                vec["dose"].append(sc["dose"])
            if "ceiling" in sc:
                vec["ceiling"].append(sc["ceiling"])
            if i % max(1, len(vigs) // 10) == 0 or i == len(vigs):
                print(f"    [{name}] {i}/{len(vigs)} elapsed {time.time()-t0:.0f}s", flush=True)
        report[name] = {"templated": vec}

    # ---- compare ------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print(" NATIVE CHAT-TEMPLATE FAIRNESS RE-RUN  (original -> templated)", flush=True)
    print("=" * 90, flush=True)
    out_json = {}
    for name, _ in targets:
        tpl = report[name]["templated"]
        line = {"templated_dose_rate": round(_rate(tpl["dose"]), 4),
                "templated_dose_n": len(tpl["dose"]),
                "templated_ceiling_rate": round(_rate(tpl["ceiling"]), 4),
                "templated_ceiling_n": len(tpl["ceiling"])}
        # original rate: 3B/7B from checkpoint vectors (+ paired McNemar);
        # Mid from the 4-way summary (rate only, no saved per-case vectors).
        if name in orig:
            od = _rate(orig[name]["dose"])
            oc = _rate(orig[name]["ceiling"])
            if len(orig[name]["dose"]) == len(tpl["dose"]):
                line["mcnemar_vs_original_dose"] = S.mcnemar_exact(
                    orig[name]["dose"], tpl["dose"])
        else:
            od = four["statistics"]["dose"]["systems"][name]["rate"]
            oc = four["statistics"]["ceiling"]["systems"][name]["rate"]
        line["original_dose_rate"] = round(od, 4)
        line["original_ceiling_rate"] = round(oc, 4)
        out_json[name] = line
        print(f"\n{name}:", flush=True)
        print(f"  dose    : original {od*100:5.1f}%  ->  templated "
              f"{_rate(tpl['dose'])*100:5.1f}%   (Δ {(_rate(tpl['dose'])-od)*100:+.1f} pts)", flush=True)
        print(f"  ceiling : original {oc*100:5.1f}%  ->  templated "
              f"{_rate(tpl['ceiling'])*100:5.1f}%   (Δ {(_rate(tpl['ceiling'])-oc)*100:+.1f} pts)", flush=True)
    print("=" * 90, flush=True)

    out = config.PROJECT_ROOT / args.out
    json.dump(out_json, open(out, "w"), indent=2)
    print(f"\n[✓] Template re-run summary -> {out}", flush=True)


if __name__ == "__main__":
    main()
