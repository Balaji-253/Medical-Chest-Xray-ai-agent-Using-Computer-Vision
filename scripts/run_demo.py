from app.agents.orchestrator import MedicalResearchAgent
from app.schemas.models import AnalysisRequest, PatientData

request = AnalysisRequest(
    patient=PatientData(
        age=54,
        sex="unknown",
        symptoms=["example symptom"],
        labs={"example_lab": 1.0}
    ),
    image_path="data/sample/sample_medical_like.png",
    question="medical AI imaging evaluation and validation"
)

result = MedicalResearchAgent().run(request)
print(result)
