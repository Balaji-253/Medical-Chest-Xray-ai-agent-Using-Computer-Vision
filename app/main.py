from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas.models import AnalysisRequest
from app.agents.orchestrator import MedicalResearchAgent
from app.ml.chest_xray_inference import ChestXrayInference
from app.explainability.gradcam import generate_gradcam


app = FastAPI(
    title="MedAgent-Vision Research API",
    version="1.2.0",
    description=(
        "Research prototype for chest X-ray computer vision "
        "and medical AI agent experimentation. "
        "Not for clinical diagnosis."
    ),
)

agent = MedicalResearchAgent()
xray_engine = ChestXrayInference()

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/artifacts",
    StaticFiles(directory=str(ARTIFACTS_DIR)),
    name="artifacts",
)
@app.get("/")
async def frontend():
    return FileResponse(
        str(Path(__file__).resolve().parent / "static" / "index.html")
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "medagent-vision",
        "model": "ResNet-18",
        "device": "cpu",
        "research_only": True,
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        return agent.run(request)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...)
):
    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    filename = file.filename or "uploaded_image"
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, BMP, or WEBP."
            ),
        )

    temporary_path = None

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 20 MB.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            temporary_path = Path(temp.name)

        result = xray_engine.predict(temporary_path)

        result["filename"] = filename
        result["endpoint"] = "/analyze-image"
        result["notice"] = (
            "Research-only output. "
            "This system is not validated for clinical diagnosis."
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {exc}",
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except Exception:
                pass


@app.post("/explain-image")
async def explain_image(
    file: UploadFile = File(...),
    class_name: str = "Effusion",
):
    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    filename = file.filename or "uploaded_image"
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, BMP, or WEBP."
            ),
        )

    valid_classes = {"Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"}

    if class_name not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown class name.",
                "valid_classes": sorted(valid_classes),
            },
        )

    temporary_path = None

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 20 MB.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            temporary_path = Path(temp.name)

        output_name = (
            f"gradcam_{class_name.lower().replace(' ', '_')}.png"
        )

        output_path = ARTIFACTS_DIR / output_name

        gradcam_result = generate_gradcam(
            image_path=temporary_path,
            class_name=class_name,
            output_path=output_path,
        )

        return {
            "model": "ResNet-18",
            "device": "cpu",
            "filename": filename,
            "class_name": class_name,
            "gradcam": gradcam_result,
            "image_url": f"/artifacts/{output_name}",
            "research_only": True,
            "notice": (
                "Grad-CAM is a model explanation visualization. "
                "It does not prove the presence or absence of disease "
                "and is not a clinical diagnostic result."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {exc}",
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except Exception:
                pass


@app.get("/research-disclaimer")
def disclaimer():
    return {
        "message": (
            "Research prototype only. "
            "Not a medical device and not for diagnosis, "
            "treatment, triage, or patient-care decisions."
        )
    }


@app.post("/agent-analyze-image")
async def agent_analyze_image(
    file: UploadFile = File(...),
    age: int = 0,
    sex: str = "unknown",
    symptoms: str = "",
    question: str = "Provide a research-oriented multimodal assessment.",
    explain_class: str | None = None,
):
    """
    Single-request multimodal research endpoint.

    Combines:
    - patient information
    - image-quality analysis
    - trained chest X-ray model
    - local evidence retrieval
    - safety gating
    - optional Grad-CAM
    - audit logging

    Research-only. Not for clinical diagnosis.
    """

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    filename = file.filename or "uploaded_image"
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, BMP, or WEBP."
            ),
        )

    if age < 0 or age > 120:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 0 and 120.",
        )

    valid_classes = {
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    }

    if explain_class is not None:
        if explain_class not in valid_classes:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unknown explain_class.",
                    "valid_classes": sorted(valid_classes),
                },
            )

    temporary_path = None

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 20 MB.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            temporary_path = Path(temp.name)

        symptom_list = [
            item.strip()
            for item in symptoms.split(",")
            if item.strip()
        ]

        request = AnalysisRequest(
            patient={
                "age": age,
                "sex": sex,
                "symptoms": symptom_list,
                "labs": {},
            },
            image_path=str(temporary_path),
            question=question,
            explain_class=explain_class,
        )

        result = agent.run(request)

        result["endpoint"] = "/agent-analyze-image"
        result["filename"] = filename
        result["notice"] = (
            "Research-only multimodal output. "
            "This system is not validated for clinical diagnosis."
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal analysis failed: {exc}",
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except Exception:
                pass
