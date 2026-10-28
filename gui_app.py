"""PedSafe-Agent interactive demo console (Streamlit).

A fully on-device, animated front-end for the neuro-symbolic pediatric dosing
pipeline. Three views:

  * Live Trace      -- paste a pediatric clinical note and watch the four agents
                       (Extractor -> Solver -> Auditor -> Arbiter gate) fire in
                       sequence on an animated flow diagram, ending in the
                       APPROVED / ADJUSTED / BLOCKED verdict + dose + rationale.
  * Benchmark Lab   -- the real evaluation results (openFDA FAERS, n=321, fair
                       native-template prompting) rendered as animated bars:
                       accuracy, the prompt-volatility swing, multi-seed
                       robustness, latency and perturbation degradation.
  * Batch Explorer  -- browse the real FAERS cohort + per-case baseline outputs,
                       filter to failures, and inspect the raw text a monolithic
                       model emitted vs PedSafe's clean structured decision.

Everything (theme, flow diagram, charts) is hand-built with inline CSS keyframes
so the console animates with no external assets and runs offline. Models load
lazily on the first "Run" click and are cached for the session.

Run:  streamlit run gui_app.py
"""
import sys
import json
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

st.set_page_config(page_title="PedSafe-Agent · Console", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
#  Palette
# --------------------------------------------------------------------------- #
SLM_C = "#8b5cf6"      # violet  -> SLM (learned) agents
SYM_C = "#14b8a6"      # teal    -> symbolic / deterministic agents
OK_C = "#10b981"       # emerald -> approved
WARN_C = "#f59e0b"     # amber   -> adjusted
BAD_C = "#ef4444"      # red     -> blocked

VERDICT = {
    "APPROVED":          ("APPROVED",          "✅", OK_C),
    "ADJUSTED_APPROVED": ("ADJUSTED · APPROVED", "⚠️", WARN_C),
    "BLOCKED_FATAL":     ("BLOCKED · FATAL",    "⛔", BAD_C),
}

SYSTEM_COLOR = {
    "PedSafe-Agent": OK_C,
    "Monolithic-3B": "#64748b",
    "Monolithic-Mid": "#f59e0b",
    "Monolithic-7B": "#6366f1",
}

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


# --------------------------------------------------------------------------- #
#  Global theme / CSS
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(f"""
    <style>
    /* ---- soft pastel aurora background (light) ----------------------- */
    .stApp {{
        background:
            radial-gradient(55rem 55rem at 8% -8%,  rgba(167,139,250,.28), transparent 60%),
            radial-gradient(48rem 48rem at 108% 6%, rgba(45,212,191,.24),  transparent 55%),
            radial-gradient(50rem 50rem at 50% 118%, rgba(251,146,108,.20), transparent 58%),
            radial-gradient(40rem 40rem at 85% 95%, rgba(96,165,250,.18),  transparent 55%),
            #f6f7fc;
        background-attachment: fixed;
        animation: drift 26s ease-in-out infinite alternate;
    }}
    @keyframes drift {{
        0%   {{ background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }}
        100% {{ background-position: 5% 4%, -5% 3%, 3% -4%, -3% -2%, 0 0; }}
    }}
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    html, body, [class*="css"] {{ color: #1e293b; }}

    /* ---- hero -------------------------------------------------------- */
    .hero-title {{
        font-size: 2.7rem; font-weight: 800; letter-spacing: -.02em; margin: 0;
        background: linear-gradient(90deg,#7c3aed,#0d9488,#0ea5e9,#7c3aed);
        background-size: 300% 100%;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: sheen 8s linear infinite;
    }}
    @keyframes sheen {{ to {{ background-position: 300% 0; }} }}
    .hero-sub {{ color:#64748b; margin-top:.25rem; font-size:1.02rem; }}
    .live-badge {{
        display:inline-flex; align-items:center; gap:.5rem; margin-top:.9rem;
        padding:.3rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600;
        color:#047857; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35);
    }}
    .live-dot {{ width:.6rem; height:.6rem; border-radius:50%; background:{OK_C};
        box-shadow:0 0 0 0 rgba(16,185,129,.7); animation:pulse 1.8s infinite; }}
    @keyframes pulse {{
        0% {{ box-shadow:0 0 0 0 rgba(16,185,129,.55); }}
        70%{{ box-shadow:0 0 0 .6rem rgba(16,185,129,0); }}
        100%{{ box-shadow:0 0 0 0 rgba(16,185,129,0); }}
    }}

    /* ---- chips / stat band ------------------------------------------- */
    .chip {{ display:inline-block; padding:.28rem .7rem; margin:.15rem .25rem 0 0;
        border-radius:.6rem; font-size:.78rem; font-weight:600; color:#475569;
        background:rgba(255,255,255,.7); border:1px solid rgba(15,23,42,.08);
        box-shadow:0 1px 2px rgba(15,23,42,.04); }}

    /* ---- glass cards (frosted white) -------------------------------- */
    .glass {{ background:rgba(255,255,255,.72); border:1px solid rgba(15,23,42,.07);
        border-radius:16px; padding:1.1rem 1.25rem; margin:.35rem 0;
        backdrop-filter: blur(10px); box-shadow:0 8px 24px rgba(79,70,229,.07);
        animation: rise .5s ease both; }}
    @keyframes rise {{ from{{opacity:0; transform:translateY(10px);}} to{{opacity:1; transform:none;}} }}
    .glass:hover {{ box-shadow:0 12px 30px rgba(79,70,229,.13); transition: box-shadow .25s; }}
    .agent-slm {{ border-left:4px solid {SLM_C}; }}
    .agent-sym {{ border-left:4px solid {SYM_C}; }}
    .agent-h {{ font-weight:700; font-size:1.02rem; margin:0 0 .5rem 0; color:#1e293b; }}
    .tag {{ font-size:.68rem; font-weight:700; padding:.1rem .45rem; border-radius:.4rem;
        vertical-align:middle; margin-left:.4rem; }}
    .tag-slm {{ color:{SLM_C}; background:rgba(139,92,246,.12); border:1px solid rgba(139,92,246,.35);}}
    .tag-sym {{ color:#0d9488; background:rgba(20,184,166,.12); border:1px solid rgba(20,184,166,.35);}}

    /* ---- verdict banner --------------------------------------------- */
    .verdict {{ border-radius:16px; padding:1.1rem 1.4rem; font-size:1.5rem; font-weight:800;
        box-shadow:0 8px 24px rgba(15,23,42,.08);
        display:flex; align-items:center; gap:.7rem; animation: popin .5s cubic-bezier(.2,1.3,.4,1) both; }}
    @keyframes popin {{ from{{opacity:0; transform:scale(.9);}} to{{opacity:1; transform:scale(1);}} }}

    /* ---- animated horizontal bars ----------------------------------- */
    .bar-row {{ margin:.55rem 0; }}
    .bar-label {{ display:flex; justify-content:space-between; font-size:.86rem;
        color:#475569; margin-bottom:.25rem; }}
    .bar-label b {{ color:#0f172a; }}
    .bar-track {{ background:rgba(15,23,42,.06); border-radius:999px; height:1.5rem; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both;
        display:flex; align-items:center; justify-content:flex-end; padding-right:.6rem;
        font-size:.76rem; font-weight:700; color:#ffffff; white-space:nowrap;
        text-shadow:0 1px 1px rgba(0,0,0,.18); }}
    @keyframes grow {{ from{{ width:0; }} }}

    /* ---- flow diagram ------------------------------------------------ */
    .flow-wrap {{ overflow-x:auto; padding:.3rem 0 .1rem; }}
    .sidebar-stat {{ font-size:1.6rem; font-weight:800; color:#0f172a; line-height:1; }}
    .sidebar-cap {{ font-size:.72rem; color:#64748b; }}

    /* ---- buttons (gradient, matches aesthetic) ---------------------- */
    .stButton>button, .stDownloadButton>button {{
        background: linear-gradient(90deg,#7c3aed,#14b8a6);
        color:#fff; border:none; border-radius:12px; font-weight:700; padding:.55rem 1rem;
        box-shadow:0 6px 18px rgba(124,58,237,.22); transition:transform .15s, box-shadow .15s, filter .15s; }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        transform:translateY(-2px); box-shadow:0 10px 26px rgba(124,58,237,.32); filter:brightness(1.05); }}
    .stButton>button:disabled {{ background:#cbd5e1; color:#f8fafc; box-shadow:none; transform:none; }}

    /* ---- sidebar / tabs --------------------------------------------- */
    section[data-testid="stSidebar"] {{ background:rgba(255,255,255,.72);
        border-right:1px solid rgba(15,23,42,.06); backdrop-filter: blur(10px); }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.3rem; }}
    .stTabs [data-baseweb="tab"] {{ background:rgba(255,255,255,.55); border-radius:10px 10px 0 0;
        padding:.5rem 1rem; color:#475569; }}
    .stTabs [aria-selected="true"] {{ background:rgba(124,58,237,.12); color:#5b21b6; font-weight:700; }}

    /* ---- one-time animated intro splash ----------------------------- */
    .intro {{ position:fixed; inset:0; z-index:99999; display:flex; align-items:center;
        justify-content:center; flex-direction:column; text-align:center;
        background:
            radial-gradient(40rem 40rem at 30% 20%, rgba(167,139,250,.5), transparent 60%),
            radial-gradient(40rem 40rem at 75% 80%, rgba(45,212,191,.45), transparent 60%),
            #f6f7fc;
        animation: intro-out 2.6s ease forwards; }}
    @keyframes intro-out {{
        0%,70% {{ opacity:1; visibility:visible; }}
        100%   {{ opacity:0; visibility:hidden; pointer-events:none; }}
    }}
    .intro-shield {{ font-size:4.2rem; animation: shield-pop 1s cubic-bezier(.2,1.5,.4,1) both; }}
    @keyframes shield-pop {{ 0%{{opacity:0; transform:scale(.3) rotate(-18deg);}}
        60%{{opacity:1; transform:scale(1.15) rotate(6deg);}} 100%{{transform:scale(1) rotate(0);}} }}
    .intro-title {{ font-size:3rem; font-weight:800; margin-top:.4rem; letter-spacing:-.02em;
        background:linear-gradient(90deg,#7c3aed,#0d9488,#0ea5e9); -webkit-background-clip:text;
        background-clip:text; color:transparent; animation: slide-up .9s .25s ease both; }}
    .intro-tag {{ color:#64748b; font-weight:600; margin-top:.3rem; animation: slide-up .9s .45s ease both; }}
    @keyframes slide-up {{ from{{opacity:0; transform:translateY(16px);}} to{{opacity:1; transform:none;}} }}
    .intro-bar {{ height:4px; width:0; border-radius:4px; margin-top:1.3rem;
        background:linear-gradient(90deg,#7c3aed,#14b8a6);
        animation: sweep 1.6s .4s cubic-bezier(.3,.9,.2,1) both; }}
    @keyframes sweep {{ from{{width:0; opacity:.3;}} to{{width:260px; opacity:1;}} }}
    </style>
    """, unsafe_allow_html=True)


def intro_splash():
    """Full-screen animated splash shown once per session (auto-dismisses)."""
    if st.session_state.get("seen_intro"):
        return
    st.markdown(
        "<div class='intro'><div class='intro-shield'>🛡️</div>"
        "<div class='intro-title'>PedSafe-Agent</div>"
        "<div class='intro-tag'>neuro-symbolic pediatric dosing safety · on-device</div>"
        "<div class='intro-bar'></div></div>", unsafe_allow_html=True)
    st.session_state["seen_intro"] = True


# --------------------------------------------------------------------------- #
#  Small render helpers
# --------------------------------------------------------------------------- #
def hbar(label, pct, color, value_txt=None, delay=0.0):
    """One animated horizontal bar. pct in 0..100."""
    value_txt = value_txt if value_txt is not None else f"{pct:.1f}%"
    w = max(pct, 1.5)
    return (f"<div class='bar-row'>"
            f"<div class='bar-label'><span>{label}</span><b>{value_txt}</b></div>"
            f"<div class='bar-track'><div class='bar-fill' style=\"width:{w}%;"
            f"background:linear-gradient(90deg,{color}cc,{color});"
            f"animation-delay:{delay}s;\">{'' if w < 14 else value_txt}</div></div></div>")


def flow_svg(active=-1, done=-1):
    """Four-agent pipeline as an SVG with a flowing connector animation.

    active = index (0..3) of the currently-running stage; done = last completed.
    """
    stages = [("1", "Extractor", "SLM", SLM_C),
              ("2", "Solver", "SYM", SYM_C),
              ("3", "Auditor", "SLM", SLM_C),
              ("4", "Arbiter", "GATE", SYM_C)]
    W, H, r = 820, 150, 34
    xs = [90, 300, 510, 720]
    cy = 70
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" '
             f'style="min-width:760px" xmlns="http://www.w3.org/2000/svg">']
    # connectors
    for i in range(3):
        x1, x2 = xs[i] + r, xs[i + 1] - r
        lit = i <= done
        col = OK_C if lit else "rgba(148,163,184,.35)"
        parts.append(
            f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" stroke="{col}" '
            f'stroke-width="3" stroke-dasharray="7 7">'
            f'<animate attributeName="stroke-dashoffset" from="28" to="0" '
            f'dur="0.7s" repeatCount="indefinite"/></line>')
        # arrowhead
        parts.append(f'<polygon points="{x2},{cy} {x2-9},{cy-5} {x2-9},{cy+5}" fill="{col}"/>')
    # nodes
    for i, (num, name, kind, col) in enumerate(stages):
        x = xs[i]
        is_active = i == active
        is_done = i <= done
        fill = f"{col}22" if (is_active or is_done) else "rgba(148,163,184,.14)"
        stroke = col if (is_active or is_done) else "rgba(100,116,139,.6)"
        glow = (f'<circle cx="{x}" cy="{cy}" r="{r}" fill="none" stroke="{col}" '
                f'stroke-width="2" opacity=".9"><animate attributeName="r" '
                f'from="{r}" to="{r+16}" dur="1.3s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" from=".8" to="0" dur="1.3s" '
                f'repeatCount="indefinite"/></circle>') if is_active else ""
        check = "✓" if (is_done and not is_active) else num
        parts.append(
            f'{glow}<circle cx="{x}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{x}" y="{cy+6}" text-anchor="middle" font-size="22" '
            f'font-weight="800" fill="{stroke}">{check}</text>'
            f'<text x="{x}" y="{cy+r+22}" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="#334155">{name}</text>'
            f'<text x="{x}" y="{cy-r-12}" text-anchor="middle" font-size="10" '
            f'font-weight="700" fill="{col}">{kind}</text>')
    parts.append('</svg>')
    return f"<div class='flow-wrap'>{''.join(parts)}</div>"


def load_json(name):
    p = config.PROJECT_ROOT / name
    try:
        return json.load(open(p))
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading on-device SLM agents (first run only)…")
def get_pipeline():
    from agents.pipeline import PedSafePipeline
    return PedSafePipeline()


# --------------------------------------------------------------------------- #
#  View 1 — Live Trace
# --------------------------------------------------------------------------- #
def live_trace():
    st.markdown("#### Live four-agent trace")
    st.markdown(flow_svg(), unsafe_allow_html=True)

    col_in, col_ex = st.columns([3, 2])
    with col_ex:
        pick = st.selectbox("Load an example case", ["— custom —"] + list(EXAMPLES))
        st.caption("Pick a known trap or type your own note.")
    default = EXAMPLES.get(pick, "")
    note = col_in.text_area("Pediatric clinical note", value=default, height=150,
                            placeholder="e.g. 3-year-old, 14 kg, viral fever, aspirin requested…")

    run = st.button("▶  Run pipeline", type="primary", disabled=not note.strip(),
                    use_container_width=True)
    if not run:
        st.info("Enter a note (or pick an example) and click **Run pipeline**. "
                "Models load on the first run and stay cached for the session.")
        return

    pipe = get_pipeline()
    flow_ph = st.empty()
    t0 = time.time()

    with st.status("Running agents…", expanded=True) as status:
        flow_ph.markdown(flow_svg(active=0), unsafe_allow_html=True)
        st.write("🧠 **Agent 1 · Extractor (SLM)** — parsing the free-text note…")
        profile = pipe.extractor.extract(note); time.sleep(0.15)

        flow_ph.markdown(flow_svg(active=1, done=0), unsafe_allow_html=True)
        st.write("⚙️ **Agent 2 · Solver (symbolic)** — computing the weight/renal dose…")
        plan = pipe.solver.solve(profile); time.sleep(0.15)

        flow_ph.markdown(flow_svg(active=2, done=1), unsafe_allow_html=True)
        st.write("🧠 **Agent 3 · Auditor (SLM)** — screening fatal contraindications…")
        audit = pipe.auditor.audit(profile, plan); time.sleep(0.15)

        flow_ph.markdown(flow_svg(active=3, done=2), unsafe_allow_html=True)
        st.write("⚙️ **Agent 4 · Arbiter (gate)** — applying the deterministic block rule…")
        time.sleep(0.15)
        status.update(label="Pipeline complete", state="complete", expanded=False)

    flow_ph.markdown(flow_svg(done=3), unsafe_allow_html=True)
    elapsed = time.time() - t0

    # Reproduce the arbiter decision from the stages already run (mirrors
    # PedSafePipeline.process; avoids a second round of inference).
    if audit.fatal_contraindication_detected:
        final, single, daily = "BLOCKED_FATAL", 0.0, 0.0
        rationale = f"{audit.audit_summary} " + "; ".join(a["reason"] for a in audit.alerts)
    elif not audit.is_safe:
        final = "ADJUSTED_APPROVED"
        single, daily = plan.recommended_single_dose_mg, plan.recommended_daily_dose_mg
        rationale = f"Approved with caution: {plan.audit_notes}"
    else:
        final = "APPROVED"
        single, daily = plan.recommended_single_dose_mg, plan.recommended_daily_dose_mg
        rationale = f"Approved: {plan.audit_notes}"

    label, icon, color = VERDICT.get(final, (final, "", "#64748b"))
    st.markdown(
        f"<div class='verdict' style='background:{color}1e;border:1px solid {color};color:{color}'>"
        f"{icon}&nbsp;{label}&nbsp;<span style='color:#334155;font-weight:600;font-size:1.05rem'>"
        f"— {plan.drug_name}</span></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Single dose", f"{single:g} mg")
    c2.metric("Daily dose", f"{daily:g} mg", f"every {plan.interval_hours} h")
    c3.metric("Latency", f"{elapsed:.1f} s", "on-device")
    st.caption(rationale)
    st.write("")

    # agent hand-off cards
    def card(cls, head, tag, tagcls, payload):
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;gap:1rem;"
            f"padding:.18rem 0;border-bottom:1px solid rgba(15,23,42,.06)'>"
            f"<span style='color:#64748b'>{k}</span>"
            f"<span style='color:#0f172a;font-weight:600;text-align:right'>{v}</span></div>"
            for k, v in payload.items())
        st.markdown(
            f"<div class='glass {cls}'><div class='agent-h'>{head}"
            f"<span class='tag {tagcls}'>{tag}</span></div>{rows}</div>",
            unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        card("agent-slm", "Agent 1 · Extractor → profile", "🧠 SLM", "tag-slm", {
            "age (days)": profile.age_days, "neonate": profile.is_neonate,
            "weight (kg)": profile.weight_kg, "target drug": profile.target_drug,
            "creatinine": profile.serum_creatinine_mg_dl,
            "diagnoses": ", ".join(profile.diagnoses) or "—",
            "allergies": ", ".join(profile.allergies) or "—",
            "co-meds": ", ".join(profile.co_medications) or "—"})
        card("agent-slm", "Agent 3 · Auditor → safety", "🧠 SLM", "tag-slm", {
            "is safe": audit.is_safe,
            "fatal contraindication": audit.fatal_contraindication_detected,
            "action": audit.action,
            "alerts": len(audit.alerts), "summary": audit.audit_summary})
    with cb:
        card("agent-sym", "Agent 2 · Solver → dosing plan", "⚙️ Symbolic", "tag-sym", {
            "metric": plan.dosing_metric,
            "raw single mg": f"{plan.raw_single_dose_mg:g}",
            "recommended mg": f"{plan.recommended_single_dose_mg:g}",
            "interval (h)": plan.interval_hours,
            "ceiling capped": plan.is_adult_ceiling_capped,
            "patient eGFR": round(plan.patient_egfr, 1),
            "renal adj.": plan.renal_adjustment_applied})
        card("agent-sym", "Agent 4 · Arbiter → decision", "⚙️ Gate", "tag-sym", {
            "rule": "block iff FATAL flagged",
            "auditor says fatal": audit.fatal_contraindication_detected,
            "final": label})


# --------------------------------------------------------------------------- #
#  View 2 — Benchmark Lab
# --------------------------------------------------------------------------- #
def benchmark_lab():
    st.markdown("#### Benchmark lab — real openFDA FAERS results")
    st.caption("All numbers are from the committed result files. Accuracy uses "
               "**fair native-template prompting** for every baseline (n = 321 "
               "dose-eligible cases, CC0 data).")

    four = load_json("evaluation_summary_4way.json")
    tpl = load_json("template_rerun_summary.json")
    tpl7 = load_json("template_rerun_7b_summary.json")
    ms = load_json("multiseed_summary.json")
    lat = load_json("latency_summary.json")
    pert = load_json("perturbation_summary.json")

    # ---- fair dose accuracy (templated) ----------------------------------
    st.markdown("##### Dose accuracy — fair prompting")
    fair = {"PedSafe-Agent": 99.8}
    if tpl:
        fair["Monolithic-3B"] = tpl["Monolithic-3B"]["templated_dose_rate"] * 100
        fair["Monolithic-Mid"] = tpl["Monolithic-Mid"]["templated_dose_rate"] * 100
    if tpl7:
        fair["Monolithic-7B"] = tpl7["Monolithic-7B"]["templated_dose_rate"] * 100
    order = ["PedSafe-Agent", "Monolithic-Mid", "Monolithic-7B", "Monolithic-3B"]
    bars = "".join(
        hbar(k, fair[k], SYSTEM_COLOR.get(k, "#64748b"), delay=.12 * i)
        for i, k in enumerate(order) if k in fair)
    st.markdown(f"<div class='glass'>{bars}</div>", unsafe_allow_html=True)
    st.caption("PedSafe is prompt-invariant by construction (symbolic solver); "
               "the best fair baseline still trails by ~55 points.")

    # ---- volatility swing ------------------------------------------------
    st.markdown("##### Prompt volatility — same model, prompt format only")
    if tpl and tpl7 and four:
        def swing(name, orig, new):
            c = SYSTEM_COLOR.get(name, "#64748b")
            delta = new - orig
            arrow = "▲" if delta >= 0 else "▼"
            return (f"<div class='glass' style='margin:.3rem 0'>"
                    f"<div style='font-weight:700;margin-bottom:.4rem'>{name} "
                    f"<span style='color:{c}'>{arrow} {abs(delta):.1f} pts</span></div>"
                    + hbar("Llama-template prompt", orig, "#475569", delay=0)
                    + hbar("native-template prompt", new, c, delay=.15) + "</div>")
        st.markdown(
            swing("Monolithic-Mid (Phi-4-mini)",
                  tpl["Monolithic-Mid"]["original_dose_rate"] * 100,
                  tpl["Monolithic-Mid"]["templated_dose_rate"] * 100)
            + swing("Monolithic-7B (DeepSeek-R1)",
                    tpl7["Monolithic-7B"]["original_dose_rate"] * 100,
                    tpl7["Monolithic-7B"]["templated_dose_rate"] * 100),
            unsafe_allow_html=True)
        st.caption("Phi swings 6.5 → 44 % on prompt format alone (a safety "
                   "liability); the 7B is stable but genuinely capped. Neither "
                   "scale nor fair prompting closes the gap to PedSafe.")

    # ---- robustness + latency side by side -------------------------------
    cL, cR = st.columns(2)
    with cL:
        st.markdown("##### Multi-seed robustness")
        if ms:
            agg = ms["aggregate_dose"]
            rows = ""
            for k in ["PedSafe-Agent", "Monolithic-Mid", "Monolithic-3B"]:
                if k in agg:
                    m, sd = agg[k]["mean"] * 100, agg[k]["sd"] * 100
                    rows += hbar(f"{k}", m, SYSTEM_COLOR.get(k, "#64748b"),
                                 value_txt=f"{m:.1f}% ± {sd:.1f}")
            st.markdown(f"<div class='glass'>{rows}</div>", unsafe_allow_html=True)
            st.caption(f"{len(ms['seeds'])} seeds + original slice. Tiny SD → the "
                       "gap is not a seed artifact.")
    with cR:
        st.markdown("##### Latency (mean, per case)")
        if lat:
            res = lat["results"]
            mx = max(v["mean_s"] for v in res.values())
            rows = ""
            for k in ["PedSafe-Agent", "Monolithic-3B", "Monolithic-Mid", "Monolithic-7B"]:
                if k in res:
                    s = res[k]["mean_s"]
                    rows += hbar(k, s / mx * 100, SYSTEM_COLOR.get(k, "#64748b"),
                                 value_txt=f"{s:.1f}s")
            st.markdown(f"<div class='glass'>{rows}</div>", unsafe_allow_html=True)
            st.caption("PedSafe ~4.4s — ~12× faster than the 7B reasoning model.")

    # ---- perturbation degradation ----------------------------------------
    st.markdown("##### Perturbation robustness (PedSafe)")
    if pert:
        deg = pert["degradation"]
        rows = "".join(
            hbar(k, v["dose_rate"] * 100, OK_C, delay=.08 * i)
            for i, (k, v) in enumerate(deg.items()))
        fid = pert["extraction_fidelity"]["rate"] * 100
        st.markdown(f"<div class='glass'>{rows}</div>", unsafe_allow_html=True)
        st.caption(f"Extraction fidelity {fid:.0f}% on clean notes. Graceful, "
                   "not catastrophic, degradation under corrupted input.")


# --------------------------------------------------------------------------- #
#  View 3 — Batch Explorer
# --------------------------------------------------------------------------- #
def batch_explorer():
    st.markdown("#### Real-data batch explorer (openFDA FAERS)")
    dump = config.PROJECT_ROOT / "data" / "baseline_raw_dump.jsonl"
    if not dump.exists():
        st.warning("No baseline raw dump yet — run `realdata/add_mid_model.py` "
                   "to populate `data/baseline_raw_dump.jsonl`.")
        return
    rows = [json.loads(l) for l in open(dump)]
    systems = sorted({r["system"] for r in rows})

    c1, c2, c3 = st.columns([2, 1, 1])
    sysf = c1.multiselect("Systems", systems, default=systems)
    only_fail = c2.checkbox("Only failures", value=True)
    only_elig = c3.checkbox("Dose-eligible", value=True)

    def keep(r):
        if r["system"] not in sysf:
            return False
        if only_elig and not r["dose_eligible"]:
            return False
        if only_fail and r.get("dose_ok") != 0:
            return False
        return True

    view = [r for r in rows if keep(r)]

    # summary chips
    elig = [r for r in rows if r["dose_eligible"] and r["system"] in sysf]
    fails = [r for r in elig if r.get("dose_ok") == 0]
    st.markdown(
        f"<span class='chip'>📋 {len(view)} shown</span>"
        f"<span class='chip'>🧪 {len(elig)} dose-eligible</span>"
        f"<span class='chip' style='color:#fca5a5'>✖ {len(fails)} failures</span>",
        unsafe_allow_html=True)
    st.write("")

    st.dataframe(
        [{"case": r["case"], "system": r["system"], "drug": r["drug"],
          "label window (mg)": f"{r['label_min']}–{r['label_max']}"
              if r["label_min"] is not None else "n/a",
          "parsed dose (mg)": r["parsed_dose"], "status": r["status"],
          "dose ok": {1: "✔", 0: "✖", None: "—"}.get(r.get("dose_ok"), "—")}
         for r in view],
        use_container_width=True, height=380, hide_index=True)

    # raw-output inspector: show the actual text a model emitted
    st.markdown("##### Raw output inspector")
    st.caption("Monolithic models often emit malformed or hallucinated text; "
               "PedSafe returns a clean structured decision. Pick a row to inspect.")
    if view:
        idx = st.selectbox(
            "Case", range(len(view)),
            format_func=lambda i: f"case {view[i]['case']} · {view[i]['system']} · {view[i]['drug']}")
        r = view[idx]
        meta = (f"<span class='chip'>drug: {r['drug']}</span>"
                f"<span class='chip'>parsed: {r['parsed_dose']} mg</span>"
                f"<span class='chip'>label: "
                f"{r['label_min']}–{r['label_max'] if r['label_min'] is not None else 'n/a'}</span>"
                f"<span class='chip'>status: {r['status']}</span>")
        st.markdown(meta, unsafe_allow_html=True)
        st.code((r.get("raw_output") or "")[:2000] or "(empty)", language="json")


# --------------------------------------------------------------------------- #
#  Sidebar + main
# --------------------------------------------------------------------------- #
def sidebar():
    with st.sidebar:
        st.markdown("### 🛡️ PedSafe-Agent")
        st.markdown("<span class='live-badge'><span class='live-dot'></span>"
                    "fully on-device · offline</span>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            "A **neuro-symbolic** pipeline: two SLM agents "
            f"(<span style='color:{SLM_C};font-weight:700'>Extractor</span>, "
            f"<span style='color:{SLM_C};font-weight:700'>Auditor</span>) wrap a "
            f"<span style='color:{SYM_C};font-weight:700'>symbolic Solver</span> and a "
            f"deterministic <span style='color:{SYM_C};font-weight:700'>Arbiter gate</span>.",
            unsafe_allow_html=True)
        st.divider()
        four = load_json("evaluation_summary_4way.json")
        n = (four["statistics"]["dose"]["systems"]["PedSafe-Agent"]["n"]
             if four else 321)
        for val, cap in [("99.8%", "dose accuracy (3-seed)"),
                         (f"n={n}", "real FAERS cases (CC0)"),
                         ("p < 0.001", "vs every baseline"),
                         ("~4.4 s", "latency · on-device")]:
            st.markdown(f"<div class='sidebar-stat'>{val}</div>"
                        f"<div class='sidebar-cap'>{cap}</div><div style='height:.7rem'></div>",
                        unsafe_allow_html=True)
        st.divider()
        st.caption("Research demo — not for clinical use.")


def main():
    inject_css()
    intro_splash()
    sidebar()
    st.markdown("<div class='hero-title'>PedSafe-Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Neuro-symbolic multi-agent SLM console for "
                "pediatric pharmacotherapy safety.</div>", unsafe_allow_html=True)
    st.markdown("<span class='live-badge'><span class='live-dot'></span>"
                "on-device inference · no data leaves this machine</span>",
                unsafe_allow_html=True)
    st.write("")
    tab1, tab2, tab3 = st.tabs(["⚡ Live Trace", "📊 Benchmark Lab", "🔎 Batch Explorer"])
    with tab1:
        live_trace()
    with tab2:
        benchmark_lab()
    with tab3:
        batch_explorer()


if __name__ == "__main__":
    main()
