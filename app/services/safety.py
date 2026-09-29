def safety_gate(vision: dict, patient: dict, evidence: list[dict]) -> tuple[float, bool, list[str]]:
    reasons = []
    uncertainty = 0.15

    if not vision.get("available"):
        uncertainty += 0.30
        reasons.append("image_missing_or_unavailable")

    quality = float(vision.get("quality_score", 0.0))
    if quality < 0.45:
        uncertainty += 0.25
        reasons.append("low_image_quality")

    if not evidence:
        uncertainty += 0.15
        reasons.append("limited_evidence")

    if "low_resolution" in vision.get("flags", []):
        uncertainty += 0.10
        reasons.append("low_resolution")

    uncertainty = min(round(uncertainty, 3), 1.0)
    review = uncertainty >= 0.45
    return uncertainty, review, reasons
