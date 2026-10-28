from typing import List, Optional
from pydantic import BaseModel


class GroundTruthDose(BaseModel):
    calculated_single_dose_mg: float
    recommended_interval_hours: int
    is_adult_ceiling_capped: bool
    is_renal_adjusted: bool
    has_fatal_contraindication: bool
    expected_action: str  # "APPROVED" or "BLOCKED_FATAL"
    fatal_reason: Optional[str] = None
    # Label-permitted single-dose range (mg) for this patient's weight, with the
    # adult ceiling applied. Used for defensible, non-circular range scoring.
    label_min_dose_mg: Optional[float] = None
    label_max_dose_mg: Optional[float] = None
    adult_ceiling_mg: Optional[float] = None
    # False when a single mg/kg range is not well defined (e.g. aspirin): the
    # case is kept for safety analysis but excluded from the dose-accuracy metric.
    dose_scoring_eligible: bool = True


class SyntheticVignette(BaseModel):
    vignette_id: str
    cohort: str  # "neonate", "infant", "child", "adolescent_obese"
    trap_category: str  # "obesity_ceiling", "renal_impairment", "neonatal_fatal", "aspirin_reye", "standard"
    clinical_narrative: str
    target_drug: str
    ground_truth: GroundTruthDose
