import time
from typing import Dict, Any, Optional
from pydantic import BaseModel
from .extractor import ParameterExtractorAgent, ExtractedPatientProfile
from .solver import SymbolicDosingSolverAgent, CalculatedDosingPlan
from .auditor import ContraindicationAuditorAgent, SafetyAuditResult
import config


class ClinicalDecisionOutput(BaseModel):
    patient_id: str
    target_drug: str
    final_status: str  # "APPROVED", "ADJUSTED_APPROVED", or "BLOCKED_FATAL"
    recommended_single_dose_mg: float
    recommended_daily_dose_mg: float
    interval_hours: int
    is_adult_ceiling_capped: bool
    renal_adjustment_applied: bool
    patient_egfr: float
    safety_alerts: list
    execution_time_seconds: float
    clinical_rationale: str


class PedSafePipeline:
    """
    End-to-end Multi-Agent SLM Pipeline for Pediatric Pharmacotherapy Safety.
    Orchestrates Extractor, Solver, and Auditor agents into a fault-tolerant decision pipeline.
    """
    def __init__(
        self,
        extractor_model: str = config.DEFAULT_EXTRACTOR_MODEL,
        auditor_model: str = config.DEFAULT_AUDITOR_MODEL,
        run_auditor: bool = True,
        enforce_gate: bool = True,
    ):
        self.extractor = ParameterExtractorAgent(model_path=extractor_model)
        self.solver = SymbolicDosingSolverAgent()
        self.auditor = ContraindicationAuditorAgent(model_path=auditor_model)
        # Ablation switches:
        #   run_auditor=False   -> skip the contraindication Auditor entirely
        #   enforce_gate=False  -> run the Auditor but do NOT let the Arbiter
        #                          block on a detected fatal contraindication
        self.run_auditor = run_auditor
        self.enforce_gate = enforce_gate

    def process(self, clinical_text: str) -> ClinicalDecisionOutput:
        start_time = time.time()

        # Step 1: Parameter Extraction (Agent 1)
        profile = self.extractor.extract(clinical_text)

        # Step 2: Symbolic Pharmacokinetic Dosing (Agent 2)
        plan = self.solver.solve(profile)

        # Step 3: Safety & Contraindication Audit (Agent 3) -- ablatable
        if self.run_auditor:
            audit = self.auditor.audit(profile, plan)
        else:
            audit = SafetyAuditResult(
                is_safe=True, fatal_contraindication_detected=False,
                alerts=[], audit_summary="[ablation] auditor disabled")

        # Step 4: Decision Synthesis & Arbiter Gate (Agent 4) -- ablatable gate
        if self.enforce_gate and audit.fatal_contraindication_detected:
            final_status = "BLOCKED_FATAL"
            single_dose = 0.0
            daily_dose = 0.0
            rationale = f"BLOCKED: {audit.audit_summary} Alerts: " + "; ".join(a['reason'] for a in audit.alerts)
        elif not audit.is_safe:
            final_status = "ADJUSTED_APPROVED"
            single_dose = plan.recommended_single_dose_mg
            daily_dose = plan.recommended_daily_dose_mg
            rationale = f"APPROVED WITH CAUTION: {plan.audit_notes} Alerts: " + "; ".join(a['reason'] for a in audit.alerts)
        else:
            final_status = "APPROVED"
            single_dose = plan.recommended_single_dose_mg
            daily_dose = plan.recommended_daily_dose_mg
            rationale = f"APPROVED: {plan.audit_notes}"

        elapsed = time.time() - start_time

        return ClinicalDecisionOutput(
            patient_id=profile.patient_id or "P-UNKNOWN",
            target_drug=profile.target_drug,
            final_status=final_status,
            recommended_single_dose_mg=single_dose,
            recommended_daily_dose_mg=daily_dose,
            interval_hours=plan.interval_hours,
            is_adult_ceiling_capped=plan.is_adult_ceiling_capped,
            renal_adjustment_applied=plan.renal_adjustment_applied,
            patient_egfr=plan.patient_egfr,
            safety_alerts=audit.alerts,
            execution_time_seconds=round(elapsed, 3),
            clinical_rationale=rationale
        )
