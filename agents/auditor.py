from typing import List, Dict, Any
from pydantic import BaseModel, Field
from tools.rules import check_contraindications
from .base import MLXModelManager
import config


class SafetyAuditResult(BaseModel):
    is_safe: bool = True
    fatal_contraindication_detected: bool = False
    alerts: List[Dict[str, str]] = Field(default_factory=list)
    action: str = "PROCEED"  # "PROCEED", "REVISE_DOSE", or "BLOCK_PRESCRIPTION"
    audit_summary: str = ""


class ContraindicationAuditorAgent:
    """
    Agent 3: Multi-tier safety and pharmacovigilance auditor.
    Evaluates patient profile and proposed dosing plan against
    fatal pediatric black-box contraindications and drug interactions.
    """
    def __init__(self, model_path: str = config.DEFAULT_AUDITOR_MODEL):
        self.model_path = model_path
        self.manager = MLXModelManager()

    def audit(self, profile, dosing_plan) -> SafetyAuditResult:
        # Step 1: Run deterministic clinical safety rules
        all_conditions = profile.diagnoses + profile.allergies
        rule_alerts = check_contraindications(
            drug_name=profile.target_drug,
            patient_conditions=all_conditions,
            age_days=profile.age_days,
            co_medications=profile.co_medications
        )

        fatal_detected = any(a.get("severity") == "FATAL" for a in rule_alerts)

        # Step 2: Formulate safety action
        if fatal_detected:
            action = "BLOCK_PRESCRIPTION"
            is_safe = False
            summary = "CRITICAL SAFETY INTERCEPTION: Lethal pediatric contraindication identified. Prescription aborted."
        elif rule_alerts:
            action = "REVISE_DOSE"
            is_safe = False
            summary = "High-risk clinical warning flagged. Dose adjustment or specialist sign-off recommended."
        else:
            action = "PROCEED"
            is_safe = True
            summary = "Safety audit cleared. No fatal contraindications or black-box interactions detected."

        return SafetyAuditResult(
            is_safe=is_safe,
            fatal_contraindication_detected=fatal_detected,
            alerts=rule_alerts,
            action=action,
            audit_summary=summary
        )
