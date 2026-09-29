from pathlib import Path
from typing import Any

from app.schemas.models import AnalysisRequest
from app.tools.vision import analyze_image
from app.tools.patient import analyze_patient
from app.tools.evidence import retrieve
from app.services.safety import safety_gate
from app.services.audit import audit
from app.ml.chest_xray_inference import ChestXrayInference
from app.explainability.gradcam import generate_gradcam


class MedicalResearchAgent:
    """
    Research-only multimodal orchestration layer.

    Combines:
    - structured patient data
    - generic image-quality analysis
    - trained chest X-ray model
    - local evidence retrieval
    - optional Grad-CAM
    - safety gating
    - audit logging
    """

    def __init__(self, xray_engine=None):
        self.xray_engine = xray_engine or ChestXrayInference()

    def run(self, request: AnalysisRequest) -> dict[str, Any]:

        # ---------------------------------------------------------
        # 1. Generic image-quality analysis
        # ---------------------------------------------------------
        vision = analyze_image(request.image_path)

        # ---------------------------------------------------------
        # 2. Structured patient-data analysis
        # ---------------------------------------------------------
        patient = analyze_patient(request.patient)

        # ---------------------------------------------------------
        # 3. Initial evidence retrieval
        # ---------------------------------------------------------
        evidence = retrieve(request.question)

        # ---------------------------------------------------------
        # 4. Real trained chest X-ray model
        # ---------------------------------------------------------
        xray = None
        explanation = None

        if (
            request.image_path
            and Path(request.image_path).exists()
        ):
            xray = self.xray_engine.predict(
                request.image_path
            )

            # Use model signals to improve local evidence retrieval.
            positive_labels = [
                item["label"]
                for item in xray.get(
                    "positive_predictions",
                    []
                )
            ]

            if positive_labels:
                evidence_query = (
                    f"{request.question} "
                    + " ".join(positive_labels)
                )

                evidence = retrieve(
                    evidence_query
                )

            # -----------------------------------------------------
            # 5. Optional Grad-CAM
            # -----------------------------------------------------
            if request.explain_class:

                output_name = (
                    "gradcam_"
                    + request.explain_class.lower()
                    .replace(" ", "_")
                    + ".png"
                )

                output_path = (
                    Path("artifacts")
                    / output_name
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                score = generate_gradcam(
                    image_path=request.image_path,
                    class_name=request.explain_class,
                    output_path=output_path,
                )

                explanation = {
                    "class_name": request.explain_class,
                    "model_score": float(score),
                    "image_url": (
                        f"/artifacts/{output_name}"
                    ),
                    "research_only": True,
                }

        # ---------------------------------------------------------
        # 6. Safety gate
        # ---------------------------------------------------------
        uncertainty, review, safety_reasons = (
            safety_gate(
                vision,
                patient,
                evidence,
            )
        )

        # ---------------------------------------------------------
        # 7. Combine model uncertainty with safety uncertainty
        # ---------------------------------------------------------
        if xray:

            model_uncertainty = {
                "LOW": 0.25,
                "MODERATE": 0.55,
                "HIGH": 0.85,
            }.get(
                xray.get("uncertainty"),
                0.85,
            )

            uncertainty = max(
                uncertainty,
                model_uncertainty,
            )

            if xray.get(
                "human_review_required"
            ):
                review = True

                safety_reasons.append(
                    "model_requests_human_review"
                )

        # Missing image always requires review.
        if not vision.get("available"):

            safety_reasons.append(
                "no_image_for_multimodal_model"
            )

            review = True

        uncertainty = min(
            round(float(uncertainty), 3),
            1.0,
        )

        # ---------------------------------------------------------
        # 8. Overall status
        # ---------------------------------------------------------
        if review:
            status = "HUMAN_REVIEW_REQUIRED"
        else:
            status = "RESEARCH_REVIEW"

        # ---------------------------------------------------------
        # 9. Research-only assessment
        # ---------------------------------------------------------
        assessment = (
            "Research-only multimodal assessment completed. "
            "Structured patient data, image-quality features, "
            "chest X-ray model output, local evidence retrieval, "
            "and safety gating were combined. "
            "The model outputs are research signals and are not "
            "a diagnosis, prognosis, treatment recommendation, "
            "or clinical decision."
        )

        if xray:
            positive_count = len(
                xray.get(
                    "positive_predictions",
                    [],
                )
            )

            assessment += (
                f" The chest X-ray model reported "
                f"{positive_count} thresholded "
                "research signals."
            )

        if safety_reasons:

            unique_reasons = list(
                dict.fromkeys(
                    safety_reasons
                )
            )

            assessment += (
                " Safety flags: "
                + ", ".join(
                    unique_reasons
                )
                + "."
            )

        # ---------------------------------------------------------
        # 10. Audit
        # ---------------------------------------------------------
        payload = {
            "status": status,
            "uncertainty": uncertainty,
            "review": review,
            "vision": vision,
            "patient": patient,
            "xray_model": xray,
            "explanation": explanation,
            "evidence_count": len(
                evidence
            ),
        }

        request_id = audit(
            "multimodal_analysis",
            payload,
        )

        # ---------------------------------------------------------
        # 11. Findings
        # ---------------------------------------------------------
        findings = [
            {
                "tool": "vision",
                "findings": vision,
            },
            {
                "tool": "patient",
                "findings": patient,
            },
            {
                "tool": "safety",
                "findings": {
                    "reasons": safety_reasons
                },
            },
        ]

        if xray is not None:

            findings.append(
                {
                    "tool": "chest_xray_model",
                    "findings": xray,
                }
            )

        if explanation is not None:

            findings.append(
                {
                    "tool": "gradcam",
                    "findings": explanation,
                }
            )

        # ---------------------------------------------------------
        # 12. Final agent response
        # ---------------------------------------------------------
        return {
            "request_id": request_id,
            "status": status,
            "research_assessment": assessment,
            "uncertainty": uncertainty,
            "human_review_required": review,
            "findings": findings,
            "evidence": evidence,
            "research_only": True,
        }
