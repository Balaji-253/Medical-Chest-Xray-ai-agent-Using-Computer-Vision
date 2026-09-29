# Project 20 — MedAgent-Vision
## Medical AI Agent + Computer Vision Research Assistant

A CPU-first research prototype combining:
- computer-vision image quality / visual feature extraction
- structured patient-data analysis
- local evidence retrieval
- agentic tool orchestration
- uncertainty / escalation logic
- human-review gate
- FastAPI API
- Docker deployment
- audit logging
- synthetic smoke-test data

### Safety
This is a research prototype, not a medical device and not a diagnostic system.
It must not be used to make clinical decisions. Never upload identifiable patient data.
The agent deliberately produces a research-oriented assessment and can escalate to human review.

## Architecture

```text
Doctor/Researcher
       |
       v
   FastAPI API
       |
       v
 Agent Orchestrator
   |      |       |
   v      v       v
Vision  Patient  Evidence
Tool    Tool     Retriever
   \      |       /
    \     |      /
     v    v     v
    Structured Findings
           |
           v
   Safety / Uncertainty
           |
     +-----+------+
     |            |
     v            v
Research Draft   Human Review
     |
     v
Audit Log
```

## Windows CPU setup

PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\generate_sample_image.py
python -m app.main
```

Git Bash:
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/generate_sample_image.py
python -m app.main
```

Open:
`http://127.0.0.1:8000/docs`

## CLI demo

```bash
python scripts/run_demo.py
```

## Test

```bash
python -m pytest -q
```

## Docker

```bash
docker compose up --build
```

## Real dataset integration

See `DATASET.md`. Do not put a downloaded clinical dataset into Git.
Start with a public/de-identified dataset, then replace the synthetic image/patient input adapter.

## Research extensions

1. Replace handcrafted image features with a pretrained medical vision encoder.
2. Add DICOM ingestion with strict de-identification.
3. Add retrieval from a licensed medical literature corpus.
4. Add a local/open-weight LLM only behind the safety layer.
5. Add calibrated uncertainty and OOD detection.
6. Evaluate patient-level leakage, external validation and subgroup performance.
7. Add MLflow/model registry and monitoring.
8. Conduct security/privacy/regulatory review before any real-world deployment.
