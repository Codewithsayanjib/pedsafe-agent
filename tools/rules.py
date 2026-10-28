from typing import Dict, Any, List, Optional

# Public Domain Pediatric Drug Knowledge Base
# Derived from open FDA package inserts and WHO Model Formulary for Children
DRUG_MONOGRAPHS: Dict[str, Dict[str, Any]] = {
    "amoxicillin": {
        "drug_name": "Amoxicillin",
        "dosing_metric": "weight",  # weight (mg/kg) or bsa (mg/m2)
        "standard_dose_mg_per_kg_per_dose": 25.0,  # e.g., 50-75 mg/kg/day divided q8h
        # Label-permitted per-dose range (mg/kg/dose) for range-based scoring.
        "min_mg_per_kg_per_dose": 13.0,  # ~40 mg/kg/day q8h
        "max_mg_per_kg_per_dose": 33.0,  # ~100 mg/kg/day q8h (high-dose otitis)
        "dose_scoring_eligible": True,
        "standard_interval_hours": 8,
        "max_single_dose_mg": 1000.0,  # Adult single dose ceiling
        "max_daily_dose_mg": 3000.0,
        "renal_clearance_dependent": True,
        "renal_rules": {
            "mild": {"egfr_threshold": 50.0, "interval_hours": 8, "dose_factor": 1.0},
            "moderate": {"egfr_threshold": 30.0, "interval_hours": 12, "dose_factor": 1.0},
            "severe": {"egfr_threshold": 10.0, "interval_hours": 24, "dose_factor": 0.5}
        },
        "contraindications": [
            {"condition": "penicillin_allergy", "severity": "FATAL", "reason": "Severe hypersensitivity / anaphylaxis risk"}
        ]
    },
    "acetaminophen": {
        "drug_name": "Acetaminophen",
        "dosing_metric": "weight",
        "standard_dose_mg_per_kg_per_dose": 15.0,
        "min_mg_per_kg_per_dose": 10.0,
        "max_mg_per_kg_per_dose": 15.0,
        "dose_scoring_eligible": True,
        "standard_interval_hours": 6,
        "max_single_dose_mg": 1000.0,  # Adult maximum single dose
        "max_daily_dose_mg": 4000.0,   # Adult daily ceiling
        "pediatric_max_daily_mg_per_kg": 75.0,
        "renal_clearance_dependent": False,
        "contraindications": [
            {"condition": "severe_hepatic_impairment", "severity": "FATAL", "reason": "Acute liver toxicity and necrosis risk"}
        ]
    },
    "ibuprofen": {
        "drug_name": "Ibuprofen",
        "dosing_metric": "weight",
        "standard_dose_mg_per_kg_per_dose": 10.0,
        "min_mg_per_kg_per_dose": 5.0,
        "max_mg_per_kg_per_dose": 10.0,
        "dose_scoring_eligible": True,
        "standard_interval_hours": 6,
        "max_single_dose_mg": 800.0,
        "max_daily_dose_mg": 2400.0,
        "renal_clearance_dependent": True,
        "contraindications": [
            {"condition": "age_under_6_months", "severity": "HIGH", "reason": "Immature renal function and elevated nephrotoxicity risk"},
            {"condition": "severe_renal_failure", "severity": "FATAL", "reason": "NSAID-induced acute tubular necrosis / exacerbation"}
        ]
    },
    "ceftriaxone": {
        "drug_name": "Ceftriaxone",
        "dosing_metric": "weight",
        "standard_dose_mg_per_kg_per_dose": 50.0,
        "min_mg_per_kg_per_dose": 50.0,
        "max_mg_per_kg_per_dose": 100.0,  # up to 100 mg/kg/day (meningitis)
        "dose_scoring_eligible": True,
        "standard_interval_hours": 24,
        "max_single_dose_mg": 2000.0,
        "max_daily_dose_mg": 4000.0,
        "renal_clearance_dependent": False,  # Biliary + renal dual elimination
        "contraindications": [
            {"condition": "neonatal_hyperbilirubinemia", "severity": "FATAL", "reason": "Displaces bilirubin from albumin leading to kernicterus"},
            {"condition": "concomitant_iv_calcium_in_neonate", "severity": "FATAL", "reason": "Lethal particulate precipitation in lungs and kidneys"}
        ]
    },
    "gentamicin": {
        "drug_name": "Gentamicin",
        "dosing_metric": "pma_weight",  # PMA-adjusted neonatal aminoglycoside dosing
        "standard_dose_mg_per_kg_per_dose": 5.0,
        "min_mg_per_kg_per_dose": 4.0,
        "max_mg_per_kg_per_dose": 7.5,  # extended-interval neonatal range
        "dose_scoring_eligible": True,
        "pma_interval_rules": [
            {"pma_max_weeks": 29.9, "interval_hours": 48},
            {"pma_max_weeks": 34.9, "interval_hours": 36},
            {"pma_max_weeks": 999.0, "interval_hours": 24}
        ],
        "max_single_dose_mg": 400.0,
        "renal_clearance_dependent": True,
        "contraindications": [
            {"condition": "aminoglycoside_hypersensitivity", "severity": "FATAL", "reason": "Anaphylaxis"},
            {"condition": "preexisting_severe_ototoxicity", "severity": "HIGH", "reason": "Irreversible cochlear/vestibular damage"}
        ]
    },
    "aspirin": {
        "drug_name": "Aspirin",
        "dosing_metric": "weight",
        "standard_dose_mg_per_kg_per_dose": 10.0,
        # Excluded from range-based dose scoring: aspirin's pediatric use spans
        # antiplatelet/Kawasaki (~1-5 mg/kg) and analgesic (~10-15 mg/kg) doses,
        # so a single permitted range is not well defined. Aspirin is retained
        # only for the safety/contraindication analysis.
        "dose_scoring_eligible": False,
        "standard_interval_hours": 6,
        "max_single_dose_mg": 650.0,
        "max_daily_dose_mg": 3000.0,
        "renal_clearance_dependent": False,
        "contraindications": [
            {"condition": "pediatric_viral_syndrome", "severity": "FATAL", "reason": "Reye's Syndrome: severe encephalopathy and hepatic lipidosis in children with varicella/influenza"}
        ]
    },
    "trimethoprim_sulfamethoxazole": {
        "drug_name": "Trimethoprim-Sulfamethoxazole",
        "dosing_metric": "weight",  # Based on TMP component
        "standard_dose_mg_per_kg_per_dose": 4.0,  # 8-10 mg TMP/kg/day divided q12h
        "standard_interval_hours": 12,
        "max_single_dose_mg": 160.0,  # 1 Double Strength tablet (160 mg TMP)
        "max_daily_dose_mg": 320.0,
        "renal_clearance_dependent": True,
        "contraindications": [
            {"condition": "age_under_2_months", "severity": "FATAL", "reason": "Kernicterus risk due to sulfonamide displacement of protein-bound bilirubin"},
            {"condition": "sulfa_allergy", "severity": "FATAL", "reason": "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis risk"}
        ]
    }
}


def get_drug_monograph(drug_name: str) -> Optional[Dict[str, Any]]:
    normalized = drug_name.lower().replace("-", "_").replace(" ", "_")
    return DRUG_MONOGRAPHS.get(normalized)


def check_contraindications(
    drug_name: str,
    patient_conditions: List[str],
    age_days: int,
    co_medications: List[str]
) -> List[Dict[str, str]]:
    """
    Evaluates patient profile against known lethal and high-risk contraindications.
    """
    monograph = get_drug_monograph(drug_name)
    if not monograph:
        return []

    alerts = []
    conditions_text = " ".join(patient_conditions).lower()
    meds_text = " ".join(co_medications).lower()
    conditions_set = {c.lower().replace(" ", "_") for c in patient_conditions}
    meds_set = {m.lower().replace(" ", "_") for m in co_medications}

    # Age specific automatic checks
    if drug_name.lower() == "ibuprofen" and age_days < 180:
        alerts.append({
            "type": "AGE_CONTRAINDICATION",
            "severity": "HIGH",
            "reason": "Ibuprofen is contraindicated in infants under 6 months due to immature renal clearance."
        })

    if "trimethoprim" in drug_name.lower() and age_days < 60:
        alerts.append({
            "type": "LETHAL_CONTRAINDICATION",
            "severity": "FATAL",
            "reason": "TMP-SMX is strictly contraindicated in infants under 2 months (Kernicterus risk)."
        })

    if drug_name.lower() == "ceftriaxone":
        if age_days <= 28 and any(term in conditions_text for term in ["hyperbilirubinemia", "jaundice", "kernicterus"]):
            alerts.append({
                "type": "LETHAL_CONTRAINDICATION",
                "severity": "FATAL",
                "reason": "Ceftriaxone is contraindicated in hyperbilirubinemic neonates (displaces bilirubin -> kernicterus)."
            })
        if age_days <= 28 and any(term in meds_text or term in conditions_text for term in ["calcium", "gluconate"]):
            alerts.append({
                "type": "LETHAL_CONTRAINDICATION",
                "severity": "FATAL",
                "reason": "Ceftriaxone + IV Calcium co-administration in neonates precipitates in lungs/kidneys and is potentially fatal."
            })

    if drug_name.lower() == "aspirin" and age_days < (19 * 365):
        if any(term in conditions_text for term in ["fever", "influenza", "varicella", "viral", "chickenpox"]):
            alerts.append({
                "type": "LETHAL_CONTRAINDICATION",
                "severity": "FATAL",
                "reason": "Aspirin administration during pediatric viral illness triggers fatal Reye's Syndrome."
            })

    # Standard monograph condition matching
    for contra in monograph.get("contraindications", []):
        cond = contra["condition"]
        if cond in conditions_set:
            alerts.append({
                "type": "CLINICAL_CONTRAINDICATION",
                "severity": contra["severity"],
                "reason": contra["reason"]
            })

    return alerts
