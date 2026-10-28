"""Dual-source label-agreement check (caveat 1: programmatic ground truth).

Our reference-standard dose windows (tools/rules.py) are derived from FDA
package inserts + the WHO Model Formulary for Children. To show those windows
are not an idiosyncratic single-source artifact, we compare each dose-eligible
drug's permitted per-dose range (mg/kg/dose) against an INDEPENDENT second
authority -- the BNF for Children (BNFc) -- and report agreement.

The BNFc values below are literature-sourced constants (standard pediatric
per-dose ranges), encoded here with provenance; this is a reference-vs-reference
concordance check, not a live scrape. For each drug we report:
  * window overlap (Jaccard = |A n B| / |A u B| of the mg/kg intervals),
  * whether each source's standard dose falls inside the other's window,
  * whether the two windows overlap at all.

Usage:  python realdata/dual_source_labels.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from tools.rules import DRUG_MONOGRAPHS

# Independent second reference: BNF for Children (BNFc) pediatric per-dose
# ranges, mg/kg/dose. Sourced from the BNFc pediatric monographs (standard
# infection / analgesic / antipyretic dosing for the 1 month-18 yr child; for
# gentamicin the neonatal extended-interval range). Values are the defensible
# low/high per-dose bounds; citation belongs in the paper's methods.
BNFC_MG_PER_KG_PER_DOSE = {
    # drug: (min, max, standard)   -- mg/kg/dose
    "amoxicillin":   (13.0, 30.0, 20.0),   # ~40-90 mg/kg/day divided q8h
    "acetaminophen": (10.0, 15.0, 15.0),   # paracetamol 10-15 mg/kg q4-6h
    "ibuprofen":     (5.0, 10.0, 10.0),    # 5-10 mg/kg tds-qds
    "ceftriaxone":   (50.0, 100.0, 80.0),  # 50-80 mg/kg od, up to 100 (meningitis)
    "gentamicin":    (4.0, 7.0, 5.0),      # neonatal extended-interval 4-7 mg/kg
}


def overlap_jaccard(a, b):
    """Jaccard overlap of two closed intervals a=(lo,hi), b=(lo,hi)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (inter / union) if union > 0 else 1.0


def main():
    rows = []
    for drug, mono in DRUG_MONOGRAPHS.items():
        if not mono.get("dose_scoring_eligible"):
            continue
        prim = (mono["min_mg_per_kg_per_dose"], mono["max_mg_per_kg_per_dose"])
        prim_std = mono["standard_dose_mg_per_kg_per_dose"]
        if drug not in BNFC_MG_PER_KG_PER_DOSE:
            rows.append({"drug": drug, "note": "no BNFc reference encoded"})
            continue
        sec_lo, sec_hi, sec_std = BNFC_MG_PER_KG_PER_DOSE[drug]
        sec = (sec_lo, sec_hi)
        jac = overlap_jaccard(prim, sec)
        rows.append({
            "drug": mono["drug_name"],
            "primary_window": list(prim), "primary_std": prim_std,
            "bnfc_window": list(sec), "bnfc_std": sec_std,
            "windows_overlap": overlap_jaccard(prim, sec) > 0,
            "jaccard": round(jac, 3),
            "prim_std_in_bnfc": sec_lo <= prim_std <= sec_hi,
            "bnfc_std_in_prim": prim[0] <= sec_std <= prim[1],
        })

    scored = [r for r in rows if "jaccard" in r]
    mean_jac = round(sum(r["jaccard"] for r in scored) / len(scored), 3)
    n_overlap = sum(1 for r in scored if r["windows_overlap"])
    n_std_both = sum(1 for r in scored if r["prim_std_in_bnfc"] and r["bnfc_std_in_prim"])

    print("=" * 96, flush=True)
    print(" DUAL-SOURCE LABEL AGREEMENT  (primary: FDA+WHO  vs  BNF for Children)", flush=True)
    print("=" * 96, flush=True)
    print(f"{'Drug':<26} | {'primary (mg/kg)':<16} | {'BNFc (mg/kg)':<14} | "
          f"{'Jaccard':>7} | {'std in both?':<12}", flush=True)
    print("-" * 96, flush=True)
    for r in scored:
        both = "yes" if (r["prim_std_in_bnfc"] and r["bnfc_std_in_prim"]) else "partial/no"
        print(f"{r['drug']:<26} | {str(r['primary_window']):<16} | "
              f"{str(r['bnfc_window']):<14} | {r['jaccard']:>7.3f} | {both:<12}", flush=True)
    print("-" * 96, flush=True)
    print(f"Drugs with overlapping windows : {n_overlap}/{len(scored)}", flush=True)
    print(f"Std dose inside BOTH windows   : {n_std_both}/{len(scored)}", flush=True)
    print(f"Mean window Jaccard overlap    : {mean_jac}", flush=True)
    print("=" * 96, flush=True)

    out = config.PROJECT_ROOT / "dual_source_label_agreement.json"
    json.dump({"primary_source": "FDA package inserts + WHO Model Formulary for Children",
               "secondary_source": "BNF for Children (BNFc)",
               "n_drugs": len(scored), "mean_jaccard": mean_jac,
               "windows_overlap": f"{n_overlap}/{len(scored)}",
               "std_in_both": f"{n_std_both}/{len(scored)}",
               "per_drug": scored}, open(out, "w"), indent=2)
    print(f"\n[✓] Dual-source agreement -> {out}", flush=True)


if __name__ == "__main__":
    main()
