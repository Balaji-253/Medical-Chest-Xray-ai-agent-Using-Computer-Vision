from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.tools.vision import analyze_image

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_vision_missing():
    result = analyze_image("does-not-exist.png")
    assert result["available"] is False
    assert "image_not_found" in result["flags"]

def test_analyze_requires_review_without_image():
    client = TestClient(app)
    payload = {
        "patient": {
            "age": 40,
            "sex": "unknown",
            "symptoms": [],
            "labs": {}
        },
        "question": "medical AI evaluation"
    }
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["human_review_required"] is True
