import json
from datetime import datetime, timezone
from uuid import uuid4
from app.config import AUDIT_FILE

def audit(event: str, payload: dict) -> str:
    request_id = str(uuid4())
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return request_id
