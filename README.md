# PedSafe-Agent — Neuro-Symbolic Multi-Agent SLM for Pediatric Dosing Safety

PedSafe-Agent is a four-agent system that turns a free-text pediatric medication
request into a safe dosing decision — **approve, dose-adjust, or block** — by
pairing two Small Language Models (SLMs) with a deterministic symbolic solver and
a hard safety gate. The thesis, validated on real adverse-event data, is that a
**small model + symbolic execution** is both *more accurate* and *more stable*
than a single larger model asked to do the arithmetic and safety reasoning itself.

Everything in this repository is runnable and every headline number is backed by
a committed result file — the `*_summary.json` files, cited next to each claim
below.

---

## 🌐 Live demo

An interactive Streamlit console — animated four-agent trace, benchmark lab, and
real-case explorer.

| Link | What works | Notes |
| :--- | :--- | :--- |
| **Hosted (always on):** https://pedsafe-agent.streamlit.app/ | Benchmark Lab + Batch Explorer | Live Trace disabled (on-device inference needs Apple MLX). |
| **On-device (temporary):** https://solution-mug-buying-featuring.trycloudflare.com | Full app incl. live GPU inference | Served from a Mac via a Cloudflare quick tunnel — reachable only while that machine runs the app; URL changes if the tunnel restarts. |

---

## 🎯 Problem

Pediatric dosing is weight- and organ-function-dependent and full of hard edges a
language model routinely gets wrong:
- **Arithmetic** — mg/kg × weight, then truncation to an adult ceiling.
- **Renal adjustment** — Bedside-Schwartz eGFR → interval extension.
- **Fatal contraindications** — e.g. aspirin in a child with a viral/varicella
  prodrome (Reye's), or ceftriaxone + IV calcium in a neonate.

A single mistake can be lethal, so a "usually right" generative model is not
acceptable. PedSafe-Agent makes the dosing arithmetic and the block rule
**deterministic**, and uses SLMs only for what they are good at: reading the note
and screening for contraindications.

---

## 🧠 Architecture — four agents

```
 free-text note
      │
      ▼
┌──────────────┐   profile    ┌──────────────┐   plan     ┌──────────────┐  safety   ┌──────────────┐
│ 1. Extractor │ ───────────▶ │ 2. Solver    │ ─────────▶ │ 3. Auditor   │ ────────▶ │ 4. Arbiter   │ ─▶ decision
│  (SLM)       │              │  (symbolic)  │            │  (SLM)       │           │  (gate)      │
└──────────────┘              └──────────────┘            └──────────────┘           └──────────────┘
```

| # | Agent | Type | Role |
| :-: | :--- | :--- | :--- |
| 1 | **Extractor** | **SLM** (Llama-3.2-3B-Instruct-4bit) | Parses the note into a validated profile (age, weight, drug, creatinine, diagnoses, allergies, co-meds). |
| 2 | **Solver** | **Symbolic** | Computes the dose from public FDA/WHO monographs: weight/BSA math, adult-ceiling truncation, Bedside-Schwartz eGFR renal adjustment. Exact by construction. |
| 3 | **Auditor** | **SLM** (Llama-3.2-3B-Instruct-4bit) | Screens the profile + plan for fatal contraindications. |
| 4 | **Arbiter** | **Deterministic gate** | Blocks iff a FATAL contraindication is flagged; otherwise approves/adjusts. No model override. |

The two **SLM** agents are Agents 1 and 3. Agents 2 and 4 are ordinary code.

---

## 📊 Dataset — real, license-free

- **Source:** openFDA **FAERS** (FDA Adverse Event Reporting System), public-domain
  **CC0** — no license or credentials required.
- **Cohort:** a deterministic pool of pediatric cases built by `realdata/`;
  evaluation runs on the first **480** cases, of which **n = 321** are
  dose-eligible (≥ the 300-sample floor for the significance tests).
- **Clinical-appropriateness filter:** overdose / hepatotoxicity / hypersensitivity
  cases are excluded from dose scoring so no system is credited for recommending a
  full dose to a patient who is already overdosed (`cohort_builder.py`).
- **Labels:** per-dose windows derived from FDA package inserts + WHO Model
  Formulary for Children (`tools/rules.py`), cross-checked against BNF for Children
  (see *dual-source agreement* below).

---

## 🏆 Headline results

**Dose/decision accuracy on real FAERS cases, with every baseline prompted using
its _own_ native chat template** (fair prompting — see *prompt volatility* below).
`n = 321` dose-eligible cases.

| System | Dose accuracy | 95% CI | vs PedSafe (McNemar, Holm-adj.) |
| :--- | :---: | :---: | :---: |
| **PedSafe-Agent** | **99.8% ± 0.1** (3-seed) · 100% (321/321) single slice | [98.8%, 100%] | — |
| Monolithic-Mid — Phi-4-mini (3.8B) | 44.2% | — | **p < 0.001** |
| Monolithic-7B — DeepSeek-R1-Distill-Qwen (7B) | 24.0% | [16.8%, 25.7%]* | **p < 0.001** |
| Monolithic-3B — Llama-3.2 (3B) | 0.9% | — | **p < 0.001** |

\* CI shown for the original-prompt 7B run; the native-template re-run moved it
only 20.9% → 24.0% (McNemar p = 0.24, not significant).

**Matched-prompt paired statistics (all four systems).** Under the identical
original-prompt condition, every baseline — including Monolithic-Mid — has a
Wilson CI and a paired McNemar vs PedSafe: 3B 2.2% [1.1–4.4%], **Mid 6.5%
[4.3–9.8%]**, 7B 20.9% [16.8–25.7%]; McNemar b = 314 / 300 / 254, c = 0 in every
case, all Holm-adjusted **p < 0.001**. → `mid_stats_parity.json`

PedSafe's dosing is produced by the **exact symbolic solver**, so on dose-eligible
cases it is correct by construction; the genuinely fallible surface is SLM
extraction, reported separately as **100% extraction fidelity** plus a perturbation
curve (below) — not a flat, unexamined 100%.

**Secondary metrics**

| Metric | PedSafe | 3B | Mid | 7B |
| :--- | :---: | :---: | :---: | :---: |
| Adult-ceiling adherence (n=59) | 98.3% | 93.2% | 40.7% | 62.7% |
| Mean latency / case | **4.4 s** | 3.7 s | 15.0 s | 53.5 s |

---

## 🔬 Supporting experiments

- **Prompt volatility (why "fair prompting" matters).** Re-scored with each
  model's native chat template, the same Phi-4-mini swings **6.5% → 44.2%** on
  prompt *format alone* — a clinical-safety liability — while the 7B is stable but
  capped (20.9% → 24.0%). PedSafe's symbolic core is prompt-invariant by
  construction. *Core finding: monolithic models are accurate-but-volatile or
  stable-but-capped; neither scale nor fair prompting closes the gap.*
  → `template_rerun_summary.json`, `template_rerun_7b_summary.json`
- **Multi-seed robustness.** Dose accuracy over 3 shuffles: PedSafe
  **99.8% ± 0.1**, Mid 6.7% ± 0.3, 3B 2.9% ± 0.6 (Llama prompt). Tiny SD → not a
  seed artifact. → `multiseed_summary.json`
- **Ablation (which agent owns safety).** On 250 synthetic cases (100 fatal),
  the full system reaches **93% fatal-block recall at 0% false-block**; removing
  *either* the Auditor or the Arbiter gate drops recall to **0%**. Agents 3+4 own
  the safety behavior. → `ablation_summary.json`
- **Perturbation robustness.** Extraction fidelity **100%** on clean notes; dose
  accuracy degrades gracefully under corrupted input (clean 100% → reorder 98.3%
  → random-case 90% → typos 73.3%, distractor 100%). → `perturbation_summary.json`
- **Dual-source label agreement.** Our FDA+WHO dose windows vs BNF for Children:
  mean Jaccard **0.941**, 5/5 windows overlap, 5/5 standard doses valid in both —
  the labels are not idiosyncratic. → `dual_source_label_agreement.json`
- **Failure taxonomy.** The monolithic failure mode is *timid*, not reckless:
  mostly over-blocking / under-dosing / no-dose, few true overdoses.
  → `failure_taxonomy.json`

---

## ⚠️ Honest limitations

- **PedSafe dose accuracy is exact-by-construction**, not a learned score; the
  fallible component is SLM extraction (reported via fidelity + perturbation).
- **No clinician adjudication** of the labels yet (public monographs + a
  second-source cross-check only).
- **Single dataset** (FAERS); one data distribution.
- **Fatal-contraindication rate** is not statistically supportable on FAERS
  (too few clean positives), so that claim rests on the **synthetic ablation** and
  qualitative case studies, which are labeled as such.

---

## 🔁 Reproduce

**On a Mac (Apple Silicon) — full system incl. live inference:**

```bash
pip install -r requirements-local.txt     # includes MLX (Apple-Silicon only)
```

```bash
python run_demo.py                         # step-through interactive demo
```

```bash
streamlit run gui_app.py                   # the live console GUI
```

Reproduce the main real-data benchmark and the headline experiments:

```bash
python evaluate.py --real --limit 480                 # 4-way FAERS benchmark + stats
python realdata/template_rerun.py --limit 480 --models 3b,mid   # fair native-template re-run
python realdata/multiseed_eval.py --limit 480 --seeds 7,13      # robustness (mean ± SD)
python realdata/perturbation_robustness.py --sample 60          # perturbation curve
python realdata/latency_benchmark.py --sample 25                # latency table
```

**Hosted demo (any OS, no GPU):** `requirements.txt` is the cloud-safe subset
(Streamlit only); the Benchmark Lab and Batch Explorer run from the committed
result files. Live Trace is disabled without MLX.

Raw run logs (`run_*.log`) are committed deliberately, so a reviewer can see the
actual per-case output and timings behind every summary.

---

## 📁 Repository structure

```
PedSafe-Agent/
├── gui_app.py                 # Streamlit demo console (3 tabs)
├── evaluate.py                # 4-way FAERS benchmark runner + significance tests
├── run_demo.py                # interactive step-through demo
├── config.py                  # model paths, physiological constants
├── agents/
│   ├── extractor.py           # Agent 1 (SLM): note → validated profile
│   ├── solver.py              # Agent 2 (symbolic): dosing engine
│   ├── auditor.py             # Agent 3 (SLM): contraindication screen
│   ├── pipeline.py            # Agent 4 (gate) + orchestration
│   └── base.py                # MLX model manager
├── tools/
│   ├── formulas.py            # Schwartz eGFR, BSA, ceiling equations
│   └── rules.py               # FDA/WHO monographs + fatal contraindications
├── realdata/                  # FAERS cohort + all experiment scripts
│   ├── stats.py               # McNemar, Wilson, bootstrap, Holm (stdlib only)
│   ├── template_rerun.py      # fair native-template re-run
│   ├── multiseed_eval.py      # robustness across seeds
│   ├── perturbation_robustness.py
│   ├── latency_benchmark.py
│   └── dual_source_labels.py
├── data/                      # FAERS cohort + per-case baseline dumps
├── *_summary.json             # committed results (one per experiment)
├── run_*.log                  # raw run logs (transparency)
├── requirements.txt           # cloud-safe (Streamlit) deps
└── requirements-local.txt     # full on-device stack (MLX)
```

---

*Research prototype — not for clinical use. Dataset: openFDA FAERS (CC0).
Models: MLX community 4-bit quantizations, run on-device via Apple Metal.*
