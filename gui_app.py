"""PedSafe-Agent interactive demo GUI (Streamlit).

Two views:
  * Live Trace    -- paste a pediatric clinical note and watch the four agents
                     (Extractor -> Solver -> Auditor -> Arbiter gate) run, with
                     the structured hand-off between each stage and the final
                     APPROVED / ADJUSTED / BLOCKED verdict + rationale.
  * Batch Explorer-- browse the real openFDA FAERS cohort and the per-case
                     baseline outputs (data/baseline_raw_dump.jsonl), filter to
                     failures, and compare each system's parsed dose vs. the
                     label window side by side.

Models load lazily (only on the first "Run" click) and are cached for the
session, so launching the GUI costs no GPU time. Fully offline / on-device.

Run:  streamlit run gui_app.py
"""
import sys
import json
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

st.set_page_config(page_title="PedSafe-Agent", page_icon="🛡️", layout="wide")

EXAMPLES = {
    "Aspirin / Reye's (fatal trap)":
        "3-year-old, 14 kg, febrile viral illness with varicella. Caregiver "
        "requesting aspirin 10 mg/kg for fever. No known allergies.",
    "Ceftriaxone + calcium neonate (fatal trap)":
        "Neonate, 6 days old, 3.2 kg, gestational age 39 weeks. Suspected "
        "sepsis. Ordered ceftriaxone; currently on IV calcium gluconate.",
    "Amoxicillin otitis (routine)":
        "2-year-old, 12 kg, acute otitis media. Prescribe amoxicillin. "
        "Creatinine 0.4 mg/dL. No allergies, no co-medications.",
    "Acetaminophen with renal impairment":
        "5-year-old, 18 kg, height 108 cm, fever. Acetaminophen requested. "
        "Serum creatinine 1.9 mg/dL (reduced renal function).",
}

STATUS_STYLE = {
    "APPROVED": ("✅", "#1a7f37"),
    "ADJUSTED_APPROVED": ("⚠️", "#9a6700"),
    "BLOCKED_FATAL": ("⛔", "#cf222e"),
}


@st.cache_resource(show_spinner="Loading on-device SLM agents (first run only)…")
def get_pipeline():
    from agents.pipeline import PedSafePipeline
    return PedSafePipeline()


def agent_card(num, title, kind, body_fn):
    tag = "🧠 SLM" if kind == "slm" else "⚙️ Symbolic"
    st.markdown(f"**Agent {num} · {title}**  &nbsp; `{tag}`", unsafe_allow_html=True)
    body_fn()
    st.divider()


def live_trace():
    st.subheader("Live four-agent trace")
    col_in, col_ex = st.columns([3, 2])
    with col_ex:
        pick = st.selectbox("Load an example", ["— custom —"] + list(EXAMPLES))
    default = EXAMPLES.get(pick, "")
    note = col_in.text_area("Pediatric clinical note", value=default, height=140,
                            placeholder="e.g. 3-year-old, 14 kg, viral fever, aspirin requested…")

    if not st.button("▶ Run pipeline", type="primary", disabled=not note.strip()):
        st.info("Enter a note (or pick an example) and click **Run pipeline**. "
                "Models load on the first run and stay cached.")
        return

    pipe = get_pipeline()

    # Run each stage explicitly so we can show the inter-agent hand-off.
    t0 = time.time()
    with st.status("Running agents…", expanded=True) as status:
        st.write("Agent 1 · Extractor (SLM) parsing the note…")
        profile = pipe.extractor.extract(note)
        st.write("Agent 2 · Solver (symbolic) computing the dose…")
        plan = pipe.solver.solve(profile)
        st.write("Agent 3 · Auditor (SLM) screening contraindications…")
        audit = pipe.auditor.audit(profile, plan)
        status.update(label="Pipeline complete", state="complete", expanded=False)
    elapsed = time.time() - t0

    # Reproduce the arbiter decision from the stages we already ran (mirrors
    # PedSafePipeline.process; avoids a second round of inference).
    if audit.fatal_contraindication_detected:
        final, single, daily = "BLOCKED_FATAL", 0.0, 0.0
        rationale = f"BLOCKED: {audit.audit_summary} " + "; ".join(a["reason"] for a in audit.alerts)
    elif not audit.is_safe:
        final = "ADJUSTED_APPROVED"
        single, daily = plan.recommended_single_dose_mg, plan.recommended_daily_dose_mg
        rationale = f"APPROVED WITH CAUTION: {plan.audit_notes}"
    else:
        final = "APPROVED"
        single, daily = plan.recommended_single_dose_mg, plan.recommended_daily_dose_mg
        rationale = f"APPROVED: {plan.audit_notes}"

    icon, color = STATUS_STYLE.get(final, ("", "#333"))
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color}1a;"
        f"border:1px solid {color};font-size:1.2rem'>"
        f"{icon} <b>{final.replace('_',' ')}</b> &nbsp;—&nbsp; {plan.drug_name}"
        f"</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Single dose", f"{single:g} mg")
    c2.metric("Daily dose", f"{daily:g} mg", f"q{plan.interval_hours}h")
    c3.metric("Latency", f"{elapsed:.1f} s")
    st.caption(rationale)
    st.divider()

    agent_card(1, "Extractor → structured profile", "slm", lambda: st.json({
        "age_days": profile.age_days, "is_neonate": profile.is_neonate,
        "weight_kg": profile.weight_kg, "target_drug": profile.target_drug,
        "serum_creatinine_mg_dl": profile.serum_creatinine_mg_dl,
        "diagnoses": profile.diagnoses, "allergies": profile.allergies,
        "co_medications": profile.co_medications}))
    agent_card(2, "Symbolic Solver → dosing plan", "sym", lambda: st.json({
        "dosing_metric": plan.dosing_metric,
        "raw_single_dose_mg": plan.raw_single_dose_mg,
        "recommended_single_dose_mg": plan.recommended_single_dose_mg,
        "interval_hours": plan.interval_hours,
        "adult_ceiling_capped": plan.is_adult_ceiling_capped,
        "patient_egfr": round(plan.patient_egfr, 1),
        "renal_adjustment_applied": plan.renal_adjustment_applied,
        "notes": plan.audit_notes}))
    agent_card(3, "Contraindication Auditor → safety result", "slm", lambda: st.json({
        "is_safe": audit.is_safe,
        "fatal_contraindication_detected": audit.fatal_contraindication_detected,
        "action": audit.action, "alerts": audit.alerts,
        "summary": audit.audit_summary}))
    agent_card(4, "Arbiter gate → final decision", "sym",
               lambda: st.write(f"Gate rule: block iff a FATAL contraindication is "
                                f"flagged. → **{final.replace('_',' ')}**"))


def batch_explorer():
    st.subheader("Real-data batch explorer (openFDA FAERS)")
    dump = config.PROJECT_ROOT / "data" / "baseline_raw_dump.jsonl"
    if not dump.exists():
        st.warning("No baseline raw dump yet — run `realdata/add_mid_model.py` "
                   "to populate `data/baseline_raw_dump.jsonl`.")
        return
    rows = [json.loads(l) for l in open(dump)]
    systems = sorted({r["system"] for r in rows})
    c1, c2 = st.columns(2)
    sysf = c1.multiselect("Systems", systems, default=systems)
    only_fail = c2.checkbox("Only dose-eligible failures", value=True)

    def is_fail(r):
        return r["dose_eligible"] and r.get("dose_ok") == 0

    view = [r for r in rows if r["system"] in sysf and (is_fail(r) if only_fail else True)]
    st.caption(f"{len(view)} rows")
    st.dataframe(
        [{"case": r["case"], "system": r["system"], "drug": r["drug"],
          "label_window_mg": f"{r['label_min']}–{r['label_max']}"
              if r["label_min"] is not None else "n/a",
          "parsed_dose_mg": r["parsed_dose"], "status": r["status"],
          "dose_ok": r.get("dose_ok")} for r in view],
        use_container_width=True, height=460)


def main():
    st.title("🛡️ PedSafe-Agent")
    st.caption("Neuro-symbolic multi-agent SLM for pediatric pharmacotherapy "
               "safety — runs fully on-device.")
    tab1, tab2 = st.tabs(["Live Trace", "Batch Explorer"])
    with tab1:
        live_trace()
    with tab2:
        batch_explorer()


if __name__ == "__main__":
    main()
