import math
from typing import Dict, Any, Optional

def calculate_schwartz_egfr(
    height_cm: float,
    serum_creatinine_mg_dl: float,
    k_constant: float = 0.413
) -> float:
    """
    Computes pediatric estimated Glomerular Filtration Rate (eGFR)
    using the Updated Bedside Schwartz Equation:
    eGFR = (k * Height in cm) / Serum Creatinine (mg/dL)
    Reference: Schwartz et al., JASN 2009.
    """
    if serum_creatinine_mg_dl <= 0:
        raise ValueError("Serum creatinine must be greater than 0.")
    if height_cm <= 0:
        raise ValueError("Height must be greater than 0.")
    
    egfr = (k_constant * height_cm) / serum_creatinine_mg_dl
    return round(egfr, 2)


def calculate_mosteller_bsa(height_cm: float, weight_kg: float) -> float:
    """
    Computes Body Surface Area (BSA) in m^2 using the Mosteller formula:
    BSA = sqrt((Height in cm * Weight in kg) / 3600)
    Reference: Mosteller RD., NEJM 1987.
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be positive.")
    
    bsa = math.sqrt((height_cm * weight_kg) / 3600.0)
    return round(bsa, 3)


def calculate_weight_based_dose(
    weight_kg: float,
    mg_per_kg_per_dose: float,
    max_single_dose_mg: float,
    daily_frequency: int = 1,
    max_daily_dose_mg: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates weight-adjusted dose with strict adult maximum ceiling enforcement.
    Formula: Dose = min(Weight * mg_per_kg, max_single_dose_mg)
    """
    if weight_kg <= 0 or mg_per_kg_per_dose <= 0:
        raise ValueError("Weight and dose rate must be positive.")
    
    raw_dose = weight_kg * mg_per_kg_per_dose
    capped = False
    final_single_dose = raw_dose

    if max_single_dose_mg is not None and raw_dose > max_single_dose_mg:
        final_single_dose = max_single_dose_mg
        capped = True

    final_daily_dose = final_single_dose * daily_frequency
    daily_capped = False
    if max_daily_dose_mg is not None and final_daily_dose > max_daily_dose_mg:
        final_daily_dose = max_daily_dose_mg
        final_single_dose = round(final_daily_dose / daily_frequency, 2)
        daily_capped = True

    return {
        "raw_single_dose_mg": round(raw_dose, 2),
        "final_single_dose_mg": round(final_single_dose, 2),
        "daily_frequency": daily_frequency,
        "final_daily_dose_mg": round(final_daily_dose, 2),
        "is_single_dose_capped": capped,
        "is_daily_dose_capped": daily_capped,
        "max_single_dose_mg": max_single_dose_mg,
        "max_daily_dose_mg": max_daily_dose_mg
    }


def calculate_bsa_based_dose(
    bsa_m2: float,
    mg_per_m2_per_dose: float,
    max_single_dose_mg: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates BSA-adjusted dose with ceiling check.
    Formula: Dose = BSA * mg_per_m2
    """
    if bsa_m2 <= 0 or mg_per_m2_per_dose <= 0:
        raise ValueError("BSA and dose rate must be positive.")
    
    raw_dose = bsa_m2 * mg_per_m2_per_dose
    capped = False
    final_dose = raw_dose

    if max_single_dose_mg is not None and raw_dose > max_single_dose_mg:
        final_dose = max_single_dose_mg
        capped = True

    return {
        "raw_single_dose_mg": round(raw_dose, 2),
        "final_single_dose_mg": round(final_dose, 2),
        "is_capped": capped,
        "max_single_dose_mg": max_single_dose_mg
    }


def calculate_postmenstrual_age_weeks(
    gestational_age_weeks: float,
    chronological_age_days: float
) -> float:
    """
    Computes Postmenstrual Age (PMA) in weeks for neonates and infants.
    PMA = Gestational Age (wks) + (Chronological Age (days) / 7)
    """
    pma = gestational_age_weeks + (chronological_age_days / 7.0)
    return round(pma, 1)
