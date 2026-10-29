# PedSafe-Agent: Neuro-Symbolic Multi-Agent SLM Architecture for Pediatric Pharmacotherapy Safety

PedSafe-Agent is a specialized multi-agent framework designed to eliminate medication and dosing errors in pediatric healthcare. It demonstrates how collaborative Small Language Models (SLMs, 1B–3B parameters) augmented with symbolic execution engines provably outperform monolithic Large Language Models (70B+ parameters) on multi-step arithmetic, physiological ceiling truncation, and fatal contraindication interception.

## 🌐 Live Demo

An interactive Streamlit console (animated four-agent trace, benchmark lab, and real-data case explorer):

- **Hosted demo (always on):** https://pedsafe-agent.streamlit.app/
  Benchmark Lab and Batch Explorer run fully in the browser. Live Trace is
  disabled here (on-device inference needs Apple MLX — see Quickstart).
- **Live on-device demo (temporary):** https://solution-mug-buying-featuring.trycloudflare.com
  Full app including live four-agent GPU inference. This link is served from a
  Mac through a Cloudflare quick tunnel, so it is only reachable while that
  machine is running the app, and the URL changes if the tunnel is restarted.

## 📁 Repository Structure

```
PedSafe-Agent/
├── config.py                 # Model paths, constants (Schwartz k=0.413, eGFR cutoffs)
├── tools/
│   ├── formulas.py           # Updated Bedside Schwartz, Mosteller BSA, ceiling equations
│   └── rules.py              # FDA/WHO pediatric monographs & lethal contraindications
├── agents/
│   ├── base.py               # MLX unified memory model manager & caching
│   ├── extractor.py          # Agent 1: Pydantic-validated clinical entity extraction
│   ├── solver.py             # Agent 2: Symbolic pharmacokinetic dosing engine
│   ├── auditor.py            # Agent 3: Deterministic & LLM-guided safety auditor
│   └── pipeline.py           # Multi-agent orchestrator & decision gatekeeper
├── benchmark/
│   ├── schema.py             # Vignette & ground truth Pydantic models
│   └── generator.py          # PediPharm procedural benchmark generator (Zero PHI)
├── evaluate.py               # Empirical evaluation runner vs. Monolithic baseline
├── run_demo.py               # Step-by-step interactive demo
└── requirements.txt          # Dependencies
```

## 🚀 Quickstart

Run with local MLX Python (Apple Silicon):

```bash
# 1. Run Interactive Demo Cases
/Users/work/miniforge3/bin/python PedSafe-Agent/run_demo.py

# 2. Run 3-Way Benchmark (PedSafe-Agent vs Monolithic 3B vs Monolithic 7B DeepSeek-R1)
/Users/work/miniforge3/bin/python PedSafe-Agent/evaluate.py --count 2

# 3. Scale up benchmark to 50 cases
/Users/work/miniforge3/bin/python PedSafe-Agent/evaluate.py --count 10
```

## 📈 Empirical Results Summary

| Evaluation Metric | Baseline 3B (Llama-3.2) | Baseline 7B (DeepSeek-R1) | PedSafe-Agent (Multi-Agent SLM) |
| :--- | :---: | :---: | :---: |
| **Exact Dosage Accuracy** | 20.0% | 40.0% | **90.0%** |
| **Adult Ceiling Cap Adherence** | 0.0% | 0.0% | **100.0%** |
| **Renal GFR Interval Adjustment** | 0.0% | 0.0% | **100.0%** |
| **Fatal Contraindication Recall** | 50.0% | 100.0% | **75.0%** |
| **Average Latency per Case** | 3.68s | 24.99s | **4.20s** |

## 📊 Benchmark Trap Categories
1. **The Obesity Ceiling Trap**: Obese children/adolescents whose raw weight-based dose exceeds adult maximums (e.g., amoxicillin > 1000 mg).
2. **The Renal Impairment Trap**: Elevated creatinine requiring Schwartz Bedside eGFR calculation and interval extension (q8h -> q12h).
3. **The Neonatal Fatal Trap**: Lethal ceftriaxone + IV calcium precipitate or hyperbilirubinemic kernicterus.
4. **The Pediatric Reye's Trap**: Aspirin prescribed in pediatric varicella/viral prodrome.
5. **Standard Controls**: Uncomplicated weight-adjusted dosing.
