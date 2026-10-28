# Real-Data Evaluation Pipeline (openFDA FAERS)

## Provenance & licensing
All data is pulled live from the **openFDA drug-event API** (`https://api.fda.gov/drug/event.json`),
which redistributes the FDA Adverse Event Reporting System (FAERS). FAERS is
**public domain / CC0** — no API key, registration, or data-use agreement is
required. This satisfies the "real dataset, no license" requirement and, unlike
MIMIC-IV/eICU, needs no credentialed PhysioNet access.

We use FAERS **only as a source of real, de-identified pediatric parameter
vectors** (age, weight, drugs, co-medications, indication, outcome). We do not
attempt to predict adverse events; FAERS is a spontaneous-reporting database and
is not a random sample of prescribing.

## Pipeline
- `openfda_faers.py` — cached, rate-limited API client + age normalization.
- `cohort_builder.py` — maps real reports onto the benchmark schema, with an
  age/weight plausibility filter and reference-grounded ground truth.
- `build_cohort.py` — CLI → `data/faers_cohort.json`.
- `stats.py` — McNemar exact test, bootstrap + Wilson CIs, Holm-Bonferroni.

Reproduce:
```bash
python realdata/build_cohort.py --per-drug 400 --out data/faers_cohort.json
```

## What the real data supports (empirical finding, 2,202-case pass)
| Cohort | n | Testable claim | Strength |
| --- | ---: | --- | --- |
| standard dosing | 1807 | exact weight-based dose arithmetic | **strong** |
| obesity/adult ceiling | 283 | ceiling-cap adherence | **strong** |
| overdose (real harm) | 104 | overdose interception vs label max | strong |
| infant NSAID | 8 | age contraindication | too few |
| named fatal contraindication (Reye's / ceftriaxone+Ca neonate) | ~0 | fatal recall | **not testable at scale** |

**Key validity finding:** FAERS does *not* contain enough clean, machine-detectable
pediatric *contraindication-context* cases to anchor a fatal-recall statistic.
Broad keyword matching produced false positives (e.g. aspirin for Kawasaki /
thrombosis prophylaxis mislabeled as Reye's); requiring a specific viral
indication removes them but leaves ~0. The real-data evaluation therefore
centers on the **arithmetic + ceiling + overdose** claims, where the symbolic
engine's advantage over monolithic SLMs is provable at n > 2000. Named fatal
contraindications are retained as (a) the synthetic/clinical-rule benchmark and
(b) qualitative real case studies from coded "Reye's syndrome" reports.
