from __future__ import annotations

import hashlib
from datetime import datetime, timezone

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
