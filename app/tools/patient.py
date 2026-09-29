from app.schemas.models import PatientData

def analyze_patient(patient: PatientData) -> dict:
    flags = []
    if patient.age >= 75:
        flags.append("older_age")
    if not patient.symptoms:
        flags.append("no_symptoms_provided")

    abnormal_lab_count = 0
    for name, value in patient.labs.items():
        if not isinstance(value, (int, float)):
            continue
        # No disease thresholds are hard-coded here. Values are only summarized.
        abnormal_lab_count += 0

    return {
        "age": patient.age,
        "sex": patient.sex,
        "symptom_count": len(patient.symptoms),
        "lab_count": len(patient.labs),
        "flags": flags,
        "clinical_interpretation": "Structured data summarized only; no diagnosis is inferred."
    }
