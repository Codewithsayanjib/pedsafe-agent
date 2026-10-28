import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.pipeline import PedSafePipeline


def demo():
    print("=" * 70)
    print(" PedSafe-Agent: Interactive Multi-Agent Clinical Demo")
    print("=" * 70)

    pipeline = PedSafePipeline()

    test_cases = [
        {
            "title": "Case 1: The Obesity Ceiling Trap (14-year-old, 85 kg)",
            "text": (
                "Patient ID: DEMO-001. A 14-year-old adolescent weighing 85.0 kg "
                "(height: 172.0 cm) presents with severe streptococcal pharyngitis. "
                "Baseline labs: serum creatinine 0.8 mg/dL. No allergies. "
                "Prescribe Amoxicillin standard pediatric regimen."
            )
        },
        {
            "title": "Case 2: Lethal Neonatal Contraindication (Ceftriaxone + IV Calcium)",
            "text": (
                "Patient ID: DEMO-002. A 12-day-old neonate weighing 3.1 kg (length 49 cm) "
                "is hospitalized in the NICU with suspected early-onset sepsis. Currently on "
                "continuous IV Calcium Gluconate infusion. Physician requests IV Ceftriaxone."
            )
        },
        {
            "title": "Case 3: Renal Impairment Adjustment (eGFR < 35)",
            "text": (
                "Patient ID: DEMO-003. A 6-year-old child weighing 20.0 kg (height 110.0 cm) "
                "with history of reflux nephropathy presents with otitis media. Recent lab: "
                "serum creatinine 1.6 mg/dL. Order Amoxicillin."
            )
        },
        {
            "title": "Case 4: Pediatric Viral Prodrome + Aspirin (Reye's Trap)",
            "text": (
                "Patient ID: DEMO-004. An 8-year-old child weighing 25.0 kg (height 122 cm) "
                "presents with chickenpox (varicella) and a fever of 39.2 C. Parents request "
                "Aspirin for pain and fever control."
            )
        }
    ]

    for case in test_cases:
        print(f"\n>>> Running {case['title']}")
        print(f"Narrative:\n{case['text']}\n")
        decision = pipeline.process(case["text"])
        
        print("-" * 50)
        print(f"  Decision Status      : {decision.final_status}")
        print(f"  Single Dose (mg)     : {decision.recommended_single_dose_mg} mg")
        print(f"  Interval             : Every {decision.interval_hours} hours")
        print(f"  Adult Cap Enforced   : {decision.is_adult_ceiling_capped}")
        print(f"  Renal Adjustment     : {decision.renal_adjustment_applied} (eGFR: {decision.patient_egfr} mL/min/1.73m2)")
        print(f"  Execution Time       : {decision.execution_time_seconds}s")
        print(f"  Rationale            : {decision.clinical_rationale}")
        print("-" * 50)


if __name__ == "__main__":
    demo()
