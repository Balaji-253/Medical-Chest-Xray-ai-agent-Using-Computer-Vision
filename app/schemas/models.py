from pydantic import BaseModel, Field
from typing import Any

class PatientData(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: str = Field(min_length=1, max_length=32)
    symptoms: list[str] = []
    labs: dict[str, float] = {}

class AnalysisRequest(BaseModel):
    patient: PatientData
    image_path: str | None = None
    question: str = "Provide a research-oriented multimodal assessment."
    explain_class: str | None = None

class ToolFinding(BaseModel):
    tool: str
    findings: dict[str, Any]

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    research_assessment: str
    uncertainty: float
    human_review_required: bool
    findings: list[ToolFinding]
    evidence: list[dict[str, Any]]
