from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
EVIDENCE_FILE = DATA_DIR / "evidence.json"
AUDIT_FILE = ARTIFACT_DIR / "audit.jsonl"

ARTIFACT_DIR.mkdir(exist_ok=True)
